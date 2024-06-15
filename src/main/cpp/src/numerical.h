/*
 * This code is customized by Weicong and Curtis to show a module to
 * handle the weight lifting part and computes the
 * formula 3 in paper "Dynamic adjustment of stimuli in real time functional
 * magnetic resonance imaging"
 */
#include <algorithm>
#include <chrono> // to calculate running time
#include <cilk/cilk.h>
#include <cilk/reducer_opadd.h>
#include <cilk/reducer_ostream.h>
#include <cmath>
#include <cstdlib>
#include <ctype.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <mkl.h>
#include <mkl_spblas.h>
#include <stdio.h>
#include <string.h>
#include <string>
#include <sys/stat.h>
#include <unordered_map>
#include <vector>
using namespace std;

#ifndef numerical_h
#define numerical_h

void clear_2d(double **mat, int row, int col);

void show_matrix(const double *mat, int row, int col);

void show_response(vector<vector<double>> response, vector<int> dimensions, int file_num, int Z, int X, int Y);

void init_mat(double *mat, int row, int col, string filepath);

void calc_inverse(double *matrix, int n);

double compute_SPRT(double *beta_hat, int col, double *C, double thetaZero, double thetaOne, double var_cBeta_hat);

vector<int> dimension_parser(const string &s);

vector<double> token_ints(string line);

void read_scan(vector<vector<double>> &response, string file_path, vector<int> &dimensions, int scan_number);

vector<vector<double>> read_all_scans(string file_path_y, vector<int> &dimensions);

vector<double> stop_boundary(double alpha, double beta);

bool check_file_existence(const string &name);

void estimate_theta1(vector<vector<double>> &res, const vector<vector<double>> &response, const vector<int> &dimensions,
                     const double *X, double **C, int num_of_C, int scan_number, int col, int second_col, double Z,
                     const vector<bool> &is_within_ROI);

double compute_cTXTX_inverse_c(double *XTX_inverse, double *c, int blk, int col, int second_col);

void write_out_voxel_level_info_to_file(const string &filename, const vector<double> &input,
                                        const vector<int> &dimensions, const bool transpose_subregion_file);

void write_out_voxel_level_info_to_file_1(const string &filename, const double *input,
                                          const vector<int> &dimensions, const bool transpose_subregion_file);

void compute_beta_hat(const double *X, const double *Y, double *result, const double *XTX_inverse,
                      const int row, const int col, const int second_col);

void compute_Z_score(double **Z_score, double **C, const double *beta_hat, double **var_cT_beta_hat,
                     int num_of_C, int pos, int col, int second_col);

void compute_XTX_inverse(double *res, const double *X, int scan_number, int col);

void compute_X_XTX_inverse(double *res, const double *X, const double *XTX_inverse, int scan_number, int col);

void compute_XTX_inverse_XT(double *res, const double *X, const double *XTX_inverse, int scan_number, int col);

void compute_H_diagnal(vector<double> &H_diagonal, const double *X_XTX_inverse, const double *X, int scan_number, int col);

void compute_R(double *res, const double *Y, const double *X, const double *beta_hat, int scan_number, int col, int second_col);

void generate_D_star_diagnal_matrix(double *D_star_diagnal, const double *R, const vector<double> &H_diagnal, int scan_number);

void generate_D_diagnal_matrix(double *D_diagnal, const double *R, int scan_number);

void generate_D_values(double *D_values, const double *R, const vector<double> &H_diagnal, int scan_number);

double compute_var_cTbeta_hat(const double *c, const double *X, const double *XTX_inverse_XT, const double *X_XTX_inverse,
                              const double *D, int scan_number, int col, int second_col);

double compute_var_cTbeta_hat_sparse(const double *c, const double *X, const double *XTX_inverse_XT, const double *X_XTX_inverse,
                                     double *D_values, int scan_number, int col, int second_col);

double compute_var_cTbeta_hat_sparse_triangle(const double *c, const double *X, const double *XTX_inverse_XT, const double *X_XTX_inverse,
                                              double *D_values, int scan_number, int col, int second_col, int n);

void assemble_new_degign_matrix(string old_path1, string old_path2, string new_path, int row, int col, int cut_point);

bool motion_correction(vector<vector<double>> &motion_data, vector<double> &FD_v, vector<double> &RMS, double threshold, int scan_number, string filename_prefix, string filename_suffix);

#endif
