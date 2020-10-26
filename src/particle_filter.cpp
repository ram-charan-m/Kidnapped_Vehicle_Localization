/** particle_filter.cpp
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 * Completed: Oct 22, 2020
 * Co-Author: Ram Charan
 */

#include "particle_filter.h"
#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>
#include <cstdlib>
#include <ctime>



using std::string;
using std::vector;
using std::normal_distribution;

std::random_device rd;
std::default_random_engine gen(rd());

// Initializes filter using GPS estimates of vehicle position
void ParticleFilter::init(double x, double y, double theta, double std[]) {
//   std::cout<<"INITIALIZATION"<<std::endl;
//   std::cout<<"GPS Vehicle Coordinates: ("<<x<<","<<y<<")"<<std::endl;
  num_particles = 100;  // The number of particles
  particles.resize(num_particles); // Reszing particles vector

  // Adding gaussian noise to position
  normal_distribution<double> dist_x(x,std[0]);
  normal_distribution<double> dist_y(y,std[1]);
  normal_distribution<double> dist_theta(theta,std[2]);
  for (int i=0; i < num_particles; ++i){
    particles[i].id = i;
    particles[i].x = dist_x(gen); 
    particles[i].y = dist_y(gen);
    particles[i].theta = dist_theta(gen);
    particles[i].weight = 1;
  } 
  is_initialized = true; // Setting initialized as true post initialization
//   std::cout<<"Particle x coordinate: "<< particles[0].x <<"Particle y coordinate: "<< particles[0].y <<std::endl;
}

// Post initialization, predicts vehicle position after the introduction of control inputs
void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {  
  // Iterating through all particles 
  std::cout<<"PREDICTION"<<std::endl;
  for (int i=0; i < num_particles; ++i){
    // Bicycle motion model for particle coordinates prediction
    
    if (abs(yaw_rate)>0.0001){
      particles[i].x += ( (velocity/yaw_rate)*( sin(particles[i].theta + (yaw_rate*delta_t)) - sin(particles[i].theta) ));
      particles[i].y += ( (velocity/yaw_rate)*( cos(particles[i].theta) - cos(particles[i].theta + (yaw_rate*delta_t)) ));
      particles[i].theta += (yaw_rate*delta_t);
    }
    else{
      particles[i].x += velocity * delta_t * cos(particles[i].theta);
      particles[i].y += velocity * delta_t * sin(particles[i].theta);
      particles[i].theta += (yaw_rate*delta_t);
    }
    std::random_device rd;
    std::default_random_engine gen(rd());
    normal_distribution<double> dist_x(0,std_pos[0]);
    normal_distribution<double> dist_y(0,std_pos[1]);
    normal_distribution<double> dist_theta(0,std_pos[2]);
    
    particles[i].x += dist_x(gen); 
    particles[i].y += dist_y(gen);
    particles[i].theta += dist_theta(gen);   
    
    std::cout<<"Predicted particle"<<i<<" coordinates are: ("<<particles[i].x<<","<<particles[i].y<<")"<<std::endl;
  } 
//   std::cout<<"Predicted Particle x coordinate:"<< particles[0].x <<"Predicted Particle y coordinate: "<< particles[0].y <<std::endl;
}



// Multivariate Gaussian density function
double ParticleFilter::multiv_prob(double sig_x, double sig_y, double x_obs, double y_obs,
                   double mu_x, double mu_y) {
  // calculate normalization term
  double gauss_norm;
  gauss_norm = 1 / (2 * M_PI * sig_x * sig_y);

  // calculate exponent
  double exponent;
  exponent = (pow(x_obs - mu_x, 2) / (2 * pow(sig_x, 2)))
    + (pow(y_obs - mu_y, 2) / (2 * pow(sig_y, 2)));

  // calculate weight using normalization terms and exponent
  double weight;
  weight = gauss_norm * exp(-exponent);

  return weight;
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {  
  
  // Iterating through every particle
  for(int i=0; i < num_particles; ++i){
    double tot_weight = 1.0;
    double x_p = particles[i].x;
    double y_p = particles[i].y;
    double theta = particles[i].theta; 
    
    // Transforming the observations from car coordinates by vehicle onboard sensors
    // to map coordinates for each particle    
    // observations in map coordinates vector
    vector<LandmarkObs> map_observations = observations;
    for(size_t j=0; j < observations.size(); ++j){
      double x_c = observations[j].x;
      double y_c = observations[j].y;
      map_observations[j].x =  x_p + (x_c*cos(theta)) - (y_c*sin(theta));
      map_observations[j].y =  y_p + (y_c*cos(theta)) + (x_c*sin(theta));
    }
    
    // Landmark Association
    for (size_t k=0; k < map_observations.size(); ++k){
      // Initializing minimum diatance from landmark to observation
      double min = 2*sensor_range; 
   
      // Finding which predicted landmark on map is closest
      for(size_t l=0; l < map_landmarks.landmark_list.size(); ++l){
        double x_lm = double(map_landmarks.landmark_list[l].x_f);
        double y_lm = double(map_landmarks.landmark_list[l].y_f);
        // Calculating distance of current particle and landmark
        double dis_P2LM = dist(x_lm, y_lm, x_p, y_p);
        if(dis_P2LM <= sensor_range){
          double dis_O2LM = dist(x_lm, y_lm, map_observations[k].x, map_observations[k].y);
          if(dis_O2LM < min){
            min = dis_O2LM;
            // Assigning the closest predicted map landmark id to observation id
            map_observations[k].id = map_landmarks.landmark_list[l].id_i;
            std::cout<<"Particle "<<i<<": Observation "<<k<<" is Associated to Landmark"<<l<<std::endl;
          }
        }
      }
    }    
    // Updating Weights
    // Computing weights for each observation and 
    // storing product of all to weight    
    for(size_t m=0; m < map_observations.size(); ++m){
      double sig_x, sig_y, x_obs, y_obs, mu_x, mu_y;
      sig_x = std_landmark[0];
      sig_y = std_landmark[1];
      x_obs = map_observations[m].x;
      y_obs = map_observations[m].y;
      // looking for landmark associated with the current observation
      for(size_t n=0; n < map_landmarks.landmark_list.size(); ++n){
        if(map_landmarks.landmark_list[n].id_i == map_observations[m].id){
          mu_x = double(map_landmarks.landmark_list[n].x_f);
          mu_y = double(map_landmarks.landmark_list[n].y_f);
//           std::cout<<"Associated Landmark is"<< n <<" for Observation "<< m <<" with coordinates ("<< mu_x <<","<< mu_y <<")"<<std::endl;
        }
      }
      tot_weight = tot_weight*multiv_prob(sig_x, sig_y, x_obs, y_obs, mu_x, mu_y);          
    }    
    particles[i].weight = tot_weight;
    std::cout<<"Total Weight for Particle "<<i<<" is "<<particles[i].weight<<std::endl;
  }
}

void ParticleFilter::resample() {
  // Weights vector  
  std::cout<<"RESAMPLING"<<std::endl;
  weights.resize(num_particles);
  for(int i=0; i < num_particles; ++i){
    weights[i]=particles[i].weight;
  }
//   double max_weight = *max_element(weights.begin(),weights.end());
  
  vector<Particle> new_particles;

//   // Starting point Index for Resampling Wheel
//   std::uniform_int_distribution<int> uniintdist(0, num_particles-1);
//   std::default_random_engine gen;
//   int index = uniintdist(gen);

//   // uniform random distribution 
//   std::uniform_real_distribution<double> unirealdist(0.0, max_weight);
//   double beta = 0.0;

//   // Resampling Wheel algorithm
//   for (int i = 0; i < num_particles; i++) {
//     beta += unirealdist(gen)*2.0;
//     while (beta > weights[index]) {
//       beta -= weights[index];
//       index = (index+1)%num_particles;
//     }
//     new_particles.push_back(particles[index]);
//   }

//   particles = new_particles;    
  
// Using Discrete_Distribution alone
  std::random_device rd;
  std::mt19937 gen(rd());
  std::discrete_distribution<int> index(weights.begin(), weights.end());

  
  for(int i=0; i < num_particles; ++i){
    int ind = index(gen);
    std::cout<<"Generated Index: "<<ind<<std::endl;
    new_particles.push_back(particles[ind]);
    std::cout<<"Resampled Particle "<<i<<" coordinates: ("<<particles[i].x<<","<<particles[i].y<<")"<<std::endl;
  }
  particles = new_particles;
}      


void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}