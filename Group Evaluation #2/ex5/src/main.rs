/*
 * Copyright 2022 Instituto Superior de Engenharia do Porto
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * 	http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

 use std::time::{Instant};

 use rayon;
 use rayon::prelude::*;
 
 use std::sync::mpsc::channel;
 use std::thread;
 use std::time;
 use threadpool::ThreadPool;

use std::fs::OpenOptions;
use serde::Serialize;
use std::error::Error;
use csv::Writer;
use csv::WriterBuilder;

#[derive(Serialize)]
struct Row{

    time_seq : f64,
    time_tp : f64,
    time_ray : f64,

}

 
  #[derive(Copy, Clone)]
  struct Complex{
    r: f64,
    i: f64
  }
 
  const NPOINTS: u32 = 1000;
  const MAXITER: u32 = 1000;
  const EPS: f64  = 1.0e-5;
  
  fn main() {
 
       /* Sequential */
      println!("Mandelbrot! Sequential...");
      let start = Instant::now();
      let num_outside_seq = get_num_points_outside_seq();
      let end = start.elapsed();
      println!("Time elapsed in sequential: {:?}", end);
      let t_seq = end;
 
      /* Threadpool */
      println!("Mandelbrot! Parallel using Threadpool...");
      let start = Instant::now();
      let num_outside_par_tp = get_num_points_outside_par_pool();
      let end = start.elapsed();
      println!("Time elapsed in parallel Threadpool: {:?}", end);
      let t_tp = end;
      
      assert!(num_outside_par_tp == num_outside_seq, "Expected num_outside to be equal: seq = {num_outside_seq}; par_threadpool = {num_outside_par_tp}");
 
      /* Rayon */
      println!("Mandelbrot! Parallel using Rayon...");
      let start = Instant::now();
      let num_outside_par_rayon = get_num_points_outside_par_rayon();
      let end = start.elapsed();
      println!("Time elapsed in parallel Rayon: {:?}", end);
      let t_ray = end;
      
      assert!(num_outside_par_rayon == num_outside_seq, "Expected num_outside to be equal: seq = {num_outside_seq}; par_rayon = {num_outside_par_rayon}");
  
      let np = NPOINTS as f64;
      let size = np*np;
      let area=2.0*2.5*1.125*((size-num_outside_par_tp as f64)/size);
      let error=area/NPOINTS as f64;
      println!("Area of Mandlebrot set = {:12.8} +/- {:12.8}\n",area,error);
  
    /*Write  */
    let row = Row{
        time_seq : t_seq.as_secs_f64(),
        time_tp : t_tp.as_secs_f64(),
        time_ray : t_ray.as_secs_f64(),

    };
    csv_write(row);
  }
  
  fn get_num_points_outside_seq() -> i32 {
  
      (0..NPOINTS).into_iter().map(|i| 
          (0..NPOINTS).into_iter().map(|j| {
              test_point(i,j)
          }).sum::<i32>()
      ).sum::<i32>()
  }
  
  
  fn get_num_points_outside_par_rayon() -> i32 {
 
      (0..NPOINTS).into_par_iter().map(|i| 
          (0..NPOINTS).into_par_iter().map(|j| {
              test_point(i,j)
          }).sum::<i32>()
      ).sum::<i32>()
      
  }
  
  fn get_num_points_outside_par_pool() -> i32 {
     let n_workers = thread::available_parallelism().unwrap().get();
     let n_jobs = 100;
     let pool = ThreadPool::new(n_workers);
 
     //println!("Created a thread pool with {:?} worker threads ", pool.max_count());
 
     let (tx, rx) = channel();
 
     for id in 0..n_jobs{
         let tx = tx.clone();
         let min = id * NPOINTS / n_jobs;
         let max = (id + 1) * NPOINTS / n_jobs;
 
         pool.execute(move || {
 
             let sum = (min..max).into_iter().map(|i| 
                 (0..NPOINTS).into_iter().map(|j| {
                     test_point(i,j)
                 }).sum::<i32>()
             ).sum::<i32>();
 
             tx.send(sum).expect("channel will be there waiting for the pool");
 
         });
     }
 
     thread::sleep(time::Duration::from_secs(1));
     //println!("There are currently {:?} worker threads active in the pool", pool.active_count());
     pool.join();
 
     let result = rx.iter().take(n_jobs as usize).fold(0, |a, b| a + b);
     //println!("Result: {}",result);
 
 
     return result;
 
 }
 
  fn test_point(i: u32, j: u32) -> i32 {
      let c = Complex{
          r: -2.0+2.5*(i as f64)/(NPOINTS as f64)+EPS,
          i: 1.125*(j as f64)/(NPOINTS as f64)+EPS
      };
  
      let mut z = c.clone();
      
      (0..MAXITER).into_iter().find(|_| {
          let temp = (z.r*z.r)-(z.i*z.i)+c.r;
          z.i = z.r*z.i*2.0+c.i;
          z.r = temp;
          z.r*z.r+z.i*z.i>4.0
      }).map_or(0, |_| 1)
 
  }

  fn csv_write(row: Row) -> Result<(), Box<dyn Error>> {

    let file = OpenOptions::new()
    .write(true)
    .append(true)
    .open("dados.csv")
    .unwrap();
    let mut wtr =WriterBuilder::new().has_headers(false).from_writer(file);
    wtr.serialize(&row)?;
    wtr.flush()?;
    Ok(())
}