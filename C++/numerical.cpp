/*
 * This code is customized by Weicong and Curtis to show a module to
 * handle the weight lifting part and computes the
 * formula 3 in paper "Dynamic adjustment of stimuli in real time functional
 * magnetic resonance imaging"
 */

#include "numerical.h"
using namespace std;

int SCAN_SIZE;
int DIMENSION_SIZE;

/*
 * Clear 2D array
 */
void clear_2d(double **mat, int row, int col) {
    for(int i=0; i<row; i++) fill_n(mat[i], col, 0.0);
}

/**
 * Param: matrix
 * Param: row dimension
 * Param: col dimension
 * Functionality: takes in an matrix and output each slot.
 */
void show_matrix(const double *mat, int row, int col) {
    //cout.precision(17);
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            cout << fixed << mat[i * col + j] << " ";
        }
        cout << endl;
    }
    cout << endl;
}

/**
  * Param: response vector which has the dimension SCAN_SIZE * [VOXEL_SIZE_1 * VOXEL_SIZE_2 * VOXEL_SIZE_2]
  * Param: file_num
  * Param: slice number
  * Param: Row number
  * Param: Col number
  * Functionality: takes in a nested vector and output each slot for select files;
  */
void show_response(vector<vector<double>> response, vector<int> dimensions, int file_num, int Z, int X, int Y){
    cout << "File number: " 
         << file_num
         << endl 
         << "Slice number: " 
         << Z << endl 
         << "Voxel location: [" 
         << X 
         << ", " 
         << Y << "]." 
         << "Value stored is :" 
         << response[file_num-1][Z * (dimensions[1] + 1) * (dimensions[2] + 1) + X * (dimensions[2] + 1) + Y]
         <<endl;
}

/*
 * Param: matrix
 * Param: row dimension
 * Param: col dimension
 * Functionality: takes in an matrix and initiate each slot with the
 * values obtained from file, as specified by file path.
 */
void init_mat(double *mat, int row, int col, string filepath) {
    ifstream data (filepath.c_str(), ios::in);
    
    if (!data) {
        cout << "File " << filepath << " could not be opened." << endl;
        exit(1);
    }
    
    for(int i = 0; i < row; i++) {
        for(int j = 0; j < col; j++) {
            data >> mat[i*col+j];
        }
    }
    
    data.close();
}

/*
 * Computes the inverse of a symmetric (Hermitian) positive-definite 
 * matrix using the Cholesky factorization
 */


void calc_inverse(double *matrix, int n){
    int info;

    /* before computing the inversion of the matrix, we need first to factor the matrix */
    dpotrf("U", &n, matrix, &n, &info);

    if(info != 0){
        cerr << "Cholesky factorization failed. Error code " << info << ". Exiting..." << endl;
        exit(0);
    }
    
    dpotri("U", &n, matrix, &n, &info);

    if(info != 0){
        cerr << "Computing inverse of matrix failed. Error code " << info <<". Exiting..." << endl;
        exit(0);
    }

    for(int i=0; i<n; i++){
        for(int j=0; j<n; j++){
            matrix[i*n+j] = matrix[j*n+i];
        }
    }
}


/* 
 * LU factorize and then compute matrix inverse using dgetri
 * Slower than Cholesky and possess some issue
 */
void calc_inverse_dgetri(double *matrix, int n){

    int* ipiv = new int[n];
    int info;
    
    // before computing the inversion of the matrix, we need first to factor the
    // matrix
    dgetrf(&n, &n, matrix, &n, ipiv, &info);

    if(info != 0){
        cerr << "LU factorization failed. Exiting..." << endl;
        exit(0);
    }
    
    double _tmp;

    /*
     * If lwork = -1, then a workspace query is assumed; 
     * the routine only calculates the optimal size of 
     * the work array, returns this value as the first entry 
     * of the work array, and no error message related to lwork 
     * is issued by xerbla.
     */
    int _lwork = -1;
    dgetri(&n, matrix, &n, ipiv, &_tmp, &_lwork, &info);
    _lwork = (int)_tmp;

    if(info != 0){
        cerr << "Calculating optimal size of the work array fails! Exiting..." << endl;
        exit(0);
    }
    
    double *work = new double[_lwork];
    dgetri(&n, matrix, &n, ipiv, work, &_lwork, &info);

    if(info != 0){
        cerr << "Calculating inverse of matrix fails! Exiting..." << endl;
        exit(0);
    }
}

/*
 * Output: SPRT result
 * Functioality: this function is to handle the weight lifting part and computes
 * the formula 3 in paper "Dynamic adjustment of stimuli in real time functional
 * magnetic resonance imaging"
 * Formula: {(c*beta_hat-theta_0)'* Var(c*beta_hat)^-1 * (c*beta_hat-theta_0) 
 *          - (c*beta_hat-theta_1)'* Var(c*beta_hat)^-1 * (c*beta_hat-theta_1)} / 2
 */
double compute_SPRT(
                double *beta_hat, int col, double *c, double thetaZero, double thetaOne, double var_cBeta_hat
                ) {
    double result = 0;
    double CB = 0; // store c * beta_hat
    for (int i = 0; i < col ; i++) {
        CB += beta_hat[i] * c[i];
    }
    return ( pow(CB - thetaZero, 2) - pow(CB - thetaOne, 2) ) / (2 * var_cBeta_hat);
}

/**
 * Param: a string in the header of each response file, e.g. in bold113.txt
 *   it is (36, 128, 128)
 *   or a string in the header of design matrix, e.g. (320, 10)
 * Output: the X-Y-Z dimension values of response file, e.g. 36, 128, 128
 *   or the row-column values of the design matrix
 * Functionality: this method takes in a string composed of dimension parameters
 *   and output three integers indicating the value of each dimension
 */
vector<int> dimension_parser(const string &s) {
    int num = 0;
    vector<int> dimensions;
    bool lastIsDigit = false;
    for (int i = 0; i < s.size(); i++) {
        if (isdigit(s[i])) {
            num *= 10;
            num += s[i] - '0';
            lastIsDigit = true;
        } else {
            if (lastIsDigit) {
                dimensions.push_back(num);
                num = 0;
                lastIsDigit = false;
            }
        }
    }
    //cout << dimensions[0] << " " << dimensions[1] << endl;
    return dimensions;
}


/**
 * This method wrote by Yi is actually incorrect. For example, say a string "1, 2, 3", 
 * it will actually parse it as numbers 1, 12, 123 instead of 1, 2, 3.
 */

/*
 * Param: a string of line containing numbers segmented by ','
 * Output: a vector of integers parsed by ','
 * Functionality: this method takes in a string, say "1,2,3" and parse it as
 * numbers separated by ',' and then return the numbers to the vector */
 
//vector<double> parse_voxel_value(string line) {
//    vector<double> ret;
//    double temp = 0;
//    for (int i = 0; i < line.size(); i++) {
//        if (isdigit(line[i])) {
//            temp *= 10;
//            temp += line[i] - '0';
//        } else {
//            ret.push_back(temp);
//        }
//    }
//    ret.push_back(temp);
//    return ret;
//}


/**
 * Param: a string of line containing numbers segmented by ','
 * Output: a vector of integers parsed by ','
 * Functionality: this method takes in a string, say "1,2,3" and parse it as
 *   numbers separated by ',' and then return the numbers to the vector
 */
vector<double> parse_voxel_value(string line) {
    istringstream ss (line);
    vector <double> ret;
    while(ss){
        string s; 
        if(!getline(ss, s, ',')) break;
        ret.push_back(stod(s));
    }
    return ret;
}

/**
 * Param: a nested vector of double type which contains the response matrix
 * Param: a string identifying the bold text file location
 * Param: a vector of int which contains the dimension information
 * Param: a counter of int to specify the latest scan number when doing real time analysis
 * Output: none
 * Functionality: For time-series issue, the single response vector is nested into 2-dimensional array.
 *    Each file is a row response and SCAN_SIZE files means SCAN_SIZE rows of response.
 */
void read_scan(vector<vector<double> > &response, string file_path, vector<int> &dimensions, int scan_number) {
    file_path+="bold"+to_string(scan_number)+".txt";
    ifstream myfile (file_path);
    string line = "";
    
    //cout << "Reading file " << file_path << endl;

    if (myfile.is_open()) {
        getline(myfile, line);
        
        /* Z - X - Y */
        dimensions = dimension_parser(line);
        
        /* get the maximum element as the base value, here (VOXEL_SIZE_2 + 1) */
        vector<double> ret (dimensions[0] * dimensions[1] * dimensions[2], 0);
        int X = 0, Y = 0, Z = -1;
        while(!myfile.eof()) {
            getline(myfile, line);
            
            /* meanning if we can find a match of word "slice" in the string */
            if (strstr(line.c_str(), "Slice") != NULL) {
                /* this is a line like "#new slice" */
                Z++;
                /* reset x axis to zero */
                X = 0;
            } else {
                /* this is a line containing numbers */
                /* reset Y to 0 */
                Y = 0;
                vector<double> line_tokens = parse_voxel_value(line);
                
                for (; Y < line_tokens.size(); Y++) {
                    // cout << "Z is" << Z << " X is " << X << " Y is " << Y <<  endl;
                    ret[Z * dimensions[1] * dimensions[2] + X * dimensions[2] + Y] = line_tokens[Y];
                }
                
                X++;
            }
        }
        /* a 2-D row here means a time-stamp */
        response.push_back(ret);
    } else {
        cout << "Can't find file " << file_path << endl;
        throw "File open error at reading matrix.";
    }
    
    myfile.close();
}

/**
 * Param: string of file path
 * Param: vector of integer which contains dimensions
 * Output: a 2-D array encoding response values
 * Functionality: this method simply runs on top of method read_scan and
 *   enumerate all possible file names and read them, store the values in the
 *   2-D response vector.
 *   Each file corresponds to the response for a certain time slot and makes up
 *   a row in 2-D array.
 */
vector<vector<double> > read_all_scans(string file_path_y, vector<int> &dimensions) {
    vector<vector<double> > response;
    for (int i = 1; i <= 100; i++) {
        string temp = file_path_y + "bold";
        
        //if (i < 10) {
        //    temp += "000";
        //    temp.append(1, i + '0');
        //} else if (i >= 10 && i <= 99) {
        //    temp += "00";
        //    string temp_second = "";
        //    int k = i;
        //    while (k > 0) {
        //        temp_second.append(1, k % 10 + '0');
        //        k /= 10;
        //    }
        //    reverse(temp_second.begin(), temp_second.end());
        //    temp += temp_second;
        //} else { // i >= 100
        //    //temp += "0";
        //    string temp_second = "";
        //    int k = i;
        //    while (k > 0) {
        //        temp_second.append(1, k % 10 + '0');
        //        k /= 10;
        //    }
        //    reverse(temp_second.begin(), temp_second.end());
        //    temp += temp_second;
        //}

        temp += to_string(i);
        temp += ".txt";
        // once the file names is generated, read contents from them
        read_scan(response, temp, dimensions, 0);
    }
    return response;
}

/**
 * Param: double -- alpha, double -- beta
 * Output: the stopping rule's boundaries [A, B]
 * Functionality: this method takes in two parameters, alpha and bate, and
 *   computes the stopping rule's boundary values A and B
 */
vector<double> stop_boundary(double alpha, double beta) {
    double A = log((1 - beta) / alpha), B = log(beta / (1 - alpha));
    vector<double> boundary;
    boundary.push_back(A);
    boundary.push_back(B);
    return boundary;
}

/*
 * Functionality: check for file existence for real-time analysis
 */
bool check_file_existence (const string& name) {
  struct stat buffer;   
  return (stat (name.c_str(), &buffer) == 0); 
}

void estimate_theta1 (vector<vector<double> >& res, const vector<vector<double> >& response, const vector<int>& dimensions, const double *X, double **C, int num_of_C, int scan_number, int col, int second_col, double Z, const vector<bool> & is_within_ROI){
    int size = dimensions[0] * dimensions[1] * dimensions[2];
    double *beta_hat = new double[col * second_col];  // 3 * 1
    double *XTX_inverse = new double[col * col];
    double **var_cT_beta_hat = new double* [num_of_C];
    for(int i=0; i<num_of_C; i++) var_cT_beta_hat[i] =  new double[size]();
    double* R = new double[scan_number];
    vector<double> H_diagnal;
    double* D_values = new double[scan_number];
    double* XTX_inverse_XT = new double[col * scan_number];
    double* X_XTX_inverse = new double[scan_number * col];
    compute_XTX_inverse(XTX_inverse, X, scan_number, col); // compute (X'X)^-1 for only once, this is required for computing beta_hat
    compute_XTX_inverse_XT(XTX_inverse_XT, X, XTX_inverse, scan_number, col);
    compute_X_XTX_inverse(X_XTX_inverse, X, XTX_inverse, scan_number, col);


    /* for each point in X-Y-Z space, compute the response array Y */
    for (int x = 0; x < dimensions[0]; x++) {
      for (int y = 0; y < dimensions[1]; y++) {
        for (int z = 0; z < dimensions[2]; z++) {
          if(is_within_ROI[x * dimensions[1] * dimensions[2] + y * dimensions[2] + z]){
            int pos = x * dimensions[1] * dimensions[2] + y * dimensions[2] + z;
            double *Y = new double[scan_number * second_col]; // Initialize the response matrix
            for (int scan = 0; scan < scan_number; scan++) {
              Y[scan] = response[scan][x * dimensions[1] * dimensions[2] + y * dimensions[2] + z];
            }

            compute_beta_hat(X, Y, beta_hat, XTX_inverse, scan_number, col, second_col);

            /* 
             * start computing var(c'beta_hat)
             */
            compute_H_diagnal(H_diagnal, X_XTX_inverse, X, scan_number, col);
            compute_R(R, Y, X, beta_hat, scan_number, col, second_col);
            generate_D_values(D_values, R, H_diagnal, scan_number);
            for(int i=0; i<num_of_C; i++){
              var_cT_beta_hat[i][pos] = compute_var_cTbeta_hat_sparse(C[i], X, XTX_inverse_XT, X_XTX_inverse, D_values, scan_number, col, second_col);
              res[i][pos] = Z * sqrt(var_cT_beta_hat[i][pos]);
            }
            delete[] Y;
          }
        }
      }
    }
    delete[] R;
    delete[] beta_hat;
    delete[] XTX_inverse;
    delete[] XTX_inverse_XT;
    delete[] X_XTX_inverse;
    delete[] D_values;
    for(int i=0; i<num_of_C; i++) delete[] var_cT_beta_hat[i];
    delete[] var_cT_beta_hat;
}

/*
 * Z = c'beta / sqrt(var(c'beta))
 */
void compute_Z_score(double** Z_score, double** C, const double* beta_hat, double** var_cT_beta_hat, int num_of_C, int pos, int col, int second_col){
  double* cT_beta_hat = new double[second_col];
  for(int i = 0; i < num_of_C; i++){
    /* compute c'beta_hat */
    cblas_dgemm(
          CblasRowMajor,
          CblasTrans,
          CblasNoTrans,
          second_col,
          second_col,
          col,
          1.0,
          C[i],
          second_col,
          beta_hat,
          second_col,
          0.0,
          cT_beta_hat,
          second_col
          );
    Z_score[i][pos] = cT_beta_hat[0] / sqrt(var_cT_beta_hat[i][pos]);
  }
  delete[] cT_beta_hat;
}

void write_out_voxel_level_info_to_file(const string& filename, const vector<double> &input, const vector<int> &dimensions, const bool transpose_subregion_file){
    ofstream myfile;
    myfile.open(filename);
    for(int x = 0; x < dimensions[0]; x++){
        for(int y = 0; y < dimensions[1]; y++){
            for(int z = 0; z < dimensions[2]; z++){
                if(transpose_subregion_file) myfile << input[x * dimensions[1] * dimensions[2] + z * dimensions[2] + y] << " ";
                else myfile << input[x * dimensions[1] * dimensions[2] + y * dimensions[2] + z] << " ";
            }
            myfile << endl;
        }
        myfile << endl;
    }
    myfile.close();
}

void write_out_voxel_level_info_to_file_1(const string& filename, const double *input, const vector<int> &dimensions, const bool transpose_subregion_file){
    ofstream myfile;
    myfile.open(filename);
    for(int x = 0; x < dimensions[0]; x++){
        for(int y = 0; y < dimensions[1]; y++){
            for(int z = 0; z < dimensions[2]; z++){
                if(transpose_subregion_file) myfile << input[x * dimensions[1] * dimensions[2] + z * dimensions[2] + y] << " ";
                else myfile << input[x * dimensions[1] * dimensions[2] + y * dimensions[2] + z] << " ";
            }
            myfile << endl;
        }
        myfile << endl;
    }
    myfile.close();
}

/*
 * Param: matrix X of size row * col
 * Param: matrix Y of size row * second_col
 * Param: matrix result of size col * col, which is passed by by reference
 * Param: matrix XTX_inverse of size col * col, which is passed back by reference 
 * Output: result contains (X'X)^-1(X'Y), which is beta_hat
 * Functionality: it takes in two matrix, X and Y, and compute
 * (X'X)^-1(X'Y) using Intel MKL library
 */
void compute_beta_hat(const double *X, const double *Y, double *result, const double *XTX_inverse, 
                      const int row, const int col, const int second_col) {

    /* keep the matrix of X'Y */
    double *XTY = new double[col * second_col];

    /* compute X'Y */
    cblas_dgemm(
                CblasRowMajor,
                CblasTrans,
                CblasNoTrans,
                col,
                second_col,
                row,
                1.0,
                X,
                col,
                Y,
                second_col,
                0.0,
                XTY,
                second_col
                );
    
    /* compute (X'X)^-1 (X'Y) */
    cblas_dgemm(
                CblasRowMajor,
                CblasNoTrans,
                CblasNoTrans,
                col,
                second_col,
                col,
                1.0,
                XTX_inverse,
                col,
                XTY,
                second_col,
                0.0,
                result,
                second_col
                );
    
    /* remember to delete dynamic memory to avoid memery leak */
    delete[] XTY;
}

/* 
 * param: easy design matrix location
 * param: hard design matrix location
 * param: new design matrix location
 * param: matrix row
 * param: matrix cut point
 * functionality: concatnate the bottom part of the new matrix to
 * first cut_point rows of the old matrix. Then write it to file.
 */
void assemble_new_degign_matrix(string old_path1, string old_path2, string new_path, int row, int col, int cut_point){
    double d;
    ifstream in(old_path1);
    ofstream out(new_path);
    for(int i=0; i<cut_point; i++){
        for(int j=0; j<col; j++){
            in >> d;
            out << d << " ";
        }
        out << endl;
    }
    in.close();
    in.open(old_path2);
    for(int i=0; i<row; i++){
        for(int j=0; j<col; j++){
            if(i<cut_point){
                in >> d;
            }
            else{
                in >> d;
                out << d << " ";
            }
        }
        if(i >= cut_point)
            out << endl;
    }
    in.close();
    out.close();
}

/*
 * Compute (X'X)^-1
 */
void compute_XTX_inverse(double* res, const double* X, int scan_number, int col){
  cblas_dgemm(
                CblasRowMajor,
                CblasTrans,
                CblasNoTrans,
                col,
                col,
                scan_number,
                1.0,
                X,
                col,
                X,
                col,
                0.0,
                res,
                col
                );
  //show_matrix(res, col, col);
  calc_inverse(res, col);
  //show_matrix(res, col, col);
}

/*
 * compute X(X'X)^-1
 */
void compute_X_XTX_inverse(double* res, const double* X, const double* XTX_inverse, int scan_number, int col){
  cblas_dgemm(
                CblasRowMajor,
                CblasNoTrans,
                CblasNoTrans,
                scan_number,
                col,
                col,
                1.0,
                X,
                col,
                XTX_inverse,
                col,
                0.0,
                res,
                col
                );
}

/*
 * compute (X'X)^-1 * X'
 */
void compute_XTX_inverse_XT(double* res, const double* X, const double* XTX_inverse, int scan_number, int col){
  cblas_dgemm(
                CblasRowMajor,
                CblasNoTrans,
                CblasTrans,
                col,
                scan_number,
                col,
                1.0,
                XTX_inverse,
                col,
                X,
                col,
                0.0,
                res,
                scan_number
                );
}

/*
 * compute H matrix and store its diagnal value into a vector
 * H = X * (X'X)^-1 * X'
 */
void compute_H_diagnal(vector<double>& H_diagonal, const double *X_XTX_inverse, const double* X, int scan_number, int col){
  H_diagonal.clear();
  double* H = new double[scan_number * scan_number];

  // compute X(X'X)^-1 * X' [scan_number, col] * [col, scan_number] = [scan_number, scan_number]
  cblas_dgemm(
                CblasRowMajor,
                CblasNoTrans,
                CblasTrans,
                scan_number,
                scan_number,
                col,
                1.0,
                X_XTX_inverse,
                col,
                X,
                col,
                0.0,
                H,
                scan_number
                );
  for(int i = 0; i < scan_number; i++){
    H_diagonal.push_back(H[i*scan_number + i]);
  }
  delete[] H;
}

/* 
 * r_i = Y_i - X * beta_hat
 * r_i is used to compute (D_r)^*
 */
void compute_R(double* res, const double* Y, const double* X, const double* beta_hat, int scan_number, int col, int second_col){
  double* Xbeta_hat = new double[scan_number * second_col];
  cblas_dgemm(
                CblasRowMajor,
                CblasNoTrans,
                CblasNoTrans,
                scan_number,
                second_col,
                col,
                1.0,
                X,
                col,
                beta_hat,
                second_col,
                0.0,
                Xbeta_hat,
                second_col
                );
  for(int i = 0; i < scan_number; i++) res[i] = Y[i] - Xbeta_hat[i];
  delete[] Xbeta_hat;
}

/*
 * update 2019-04-12: Deprecated due to slowness
 * generate D matrix and store its diagnal elements into a vector
 */
void generate_D_diagnal_matrix(double* D_diagnal, const double* R, int scan_number){
  for(int i = 0; i < scan_number; i++){
    D_diagnal[i * scan_number + i] = pow(R[i], 2);
  }
}

/*
 * update 2019-04-12: Deprecated due to slowness
 * generate D^* matrix and store its diagnal elements into a vector
 */
void generate_D_star_diagnal_matrix(double* D_star_diagnal, const double* R, const vector<double>& H_diagnal, int scan_number){
  for(int i = 0; i < scan_number; i++){
    D_star_diagnal[i * scan_number + i] = pow(R[i], 2) / (1 - H_diagnal[i]);
  }
}

/* 
 * D_values is a single vector is used as a parameter of sparse matrix computation
 */
void generate_D_values(double* D_values, const double* R, const vector<double>& H_diagnal, int scan_number){
    //int count = 0;
    for(int i = 0; i < scan_number; i++){
        //if(abs(1-H_diagnal[i]) < 0.1) count++;
        D_values[i] = pow(R[i], 2) / (1 - H_diagnal[i]);
    }
    //cout << "1-h[ii] too small: " << count << endl;
}

/*
 * update 2019-04-12: Deprecated due to slowness, use the sparse version instead
 * var(c_beta_hat) = c'(X'X)^(-1)X'D_r^(*)X(X'X)^(-1)c
 */
double compute_var_cTbeta_hat(const double* c, const double* X, const double* XTX_inverse_XT, const double* X_XTX_inverse, 
                              const double* D, int scan_number, int col, int second_col){
  double* cT_XTX_inverse_XT = new double[second_col * scan_number];
  double* cT_XTX_inverse_XT_D = new double[second_col * scan_number];
  double* cT_XTX_inverse_XT_D_X_XTX_inverse = new double[second_col * col];
  double* cT_XTX_inverse_XT_D_X_XTX_inverse_c = new double[second_col]; 

  // compute c'(X'X)^(-1)X
  cblas_dgemm(
                CblasRowMajor,
                CblasTrans,
                CblasNoTrans,
                second_col,
                scan_number,
                col,
                1.0,
                c,
                second_col,
                XTX_inverse_XT,
                scan_number,
                0.0,
                cT_XTX_inverse_XT,
                scan_number
                );

  // compute c'(X'X)^(-1)XD_r^(*)
  cblas_dgemm(
                CblasRowMajor,
                CblasNoTrans,
                CblasNoTrans,
                second_col,
                scan_number,
                scan_number,
                1.0,
                cT_XTX_inverse_XT,
                scan_number,
                D,
                scan_number,
                0.0,
                cT_XTX_inverse_XT_D,
                scan_number
                );

  // compute c'(X'X)^(-1)X'D_r^(*)X(X'X)^(-1)
  cblas_dgemm(
                CblasRowMajor,
                CblasNoTrans,
                CblasNoTrans,
                second_col,
                col,
                scan_number,
                1.0,
                cT_XTX_inverse_XT_D,
                scan_number,
                X_XTX_inverse,
                col,
                0.0,
                cT_XTX_inverse_XT_D_X_XTX_inverse,
                col
                );

  // compute c'(X'X)^(-1)X'D_r^(*)X(X'X)^(-1)c
  cblas_dgemm(
                CblasRowMajor,
                CblasNoTrans,
                CblasNoTrans,
                second_col,
                second_col,
                col,
                1.0,
                cT_XTX_inverse_XT_D_X_XTX_inverse,
                col,
                c,
                second_col,
                0.0,
                cT_XTX_inverse_XT_D_X_XTX_inverse_c,
                second_col
                );

  double ret = cT_XTX_inverse_XT_D_X_XTX_inverse_c[0];

  delete[] cT_XTX_inverse_XT;
  delete[] cT_XTX_inverse_XT_D;
  delete[] cT_XTX_inverse_XT_D_X_XTX_inverse;
  delete[] cT_XTX_inverse_XT_D_X_XTX_inverse_c;

  return ret;
}

/* 
 * use Inspector-Executor Sparse BLAS routines for fast sparse matrix computation
 * ~200 timms faster than without using spase blas routines.
 */
double compute_var_cTbeta_hat_sparse(const double* c, const double* X, const double* XTX_inverse_XT, const double* X_XTX_inverse, 
                                     double* D_values, int scan_number, int col, int second_col){
  double* cT_XTX_inverse_XT = new double[second_col * scan_number];
  double* D_X_XTX_inverse = new double[scan_number * col];
  double* cT_XTX_inverse_XT_D_X_XTX_inverse = new double[second_col * col];
  double* cT_XTX_inverse_XT_D_X_XTX_inverse_c = new double[second_col]; 

  // compute c'(X'X)^(-1)X'
  cblas_dgemm(
                CblasRowMajor,
                CblasTrans,
                CblasNoTrans,
                second_col,
                scan_number,
                col,
                1.0,
                c,
                second_col,
                XTX_inverse_XT,
                scan_number,
                0.0,
                cT_XTX_inverse_XT,
                scan_number
                );


  // reference: https://software.intel.com/en-us/mkl-developer-reference-c-mkl-sparse-create-csr
  int* rows_start = new int[scan_number];
  int* rows_end = new int[scan_number];
  int* col_indx = new int[scan_number];
  for(int i = 0; i < scan_number; i++){
    rows_start[i] = i;
    rows_end[i] = i+1;
    col_indx[i] = i;
  }

  sparse_matrix_t D;
  matrix_descr descr;
  descr.type = sparse_matrix_type_t::SPARSE_MATRIX_TYPE_DIAGONAL;
  descr.diag = sparse_diag_type_t::SPARSE_DIAG_NON_UNIT;

  mkl_sparse_d_create_csr(&D, SPARSE_INDEX_BASE_ZERO, scan_number, scan_number, rows_start, rows_end, col_indx, D_values);

  /* 
   * These optimization rules will actually make its performance worse than normal
   *
   */
  // reference: https://software.intel.com/en-us/mkl-developer-reference-c-mkl-sparse-set-mm-hint
  //mkl_sparse_set_mm_hint(D, SPARSE_OPERATION_NON_TRANSPOSE, descr, SPARSE_LAYOUT_ROW_MAJOR, scan_number, 1);

  // reference: https://software.intel.com/en-us/mkl-developer-reference-c-mkl-sparse-set-memory-hint
  //mkl_sparse_set_memory_hint(D, SPARSE_MEMORY_AGGRESSIVE); 

  // reference: https://software.intel.com/en-us/mkl-developer-reference-c-mkl-sparse-optimize
  //mkl_sparse_optimize(D);

  // reference: https://software.intel.com/en-us/mkl-developer-reference-c-mkl-sparse-mm
  // compute D_r^(*)X(X'X)^(-1)
  mkl_sparse_d_mm(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, D, descr, SPARSE_LAYOUT_ROW_MAJOR, X_XTX_inverse, col, col, 0.0, D_X_XTX_inverse, col);

  // reference: https://software.intel.com/en-us/mkl-developer-reference-c-mkl-sparse-destroy
  mkl_sparse_destroy(D);

  // compute c'(X'X)^(-1)X'D_r^(*)X(X'X)^(-1)
  cblas_dgemm(
                CblasRowMajor,
                CblasNoTrans,
                CblasNoTrans,
                second_col,
                col,
                scan_number,
                1.0,
                cT_XTX_inverse_XT,
                scan_number,
                D_X_XTX_inverse,
                col,
                0.0,
                cT_XTX_inverse_XT_D_X_XTX_inverse,
                col
                );

  // compute c'(X'X)^(-1)X'D_r^(*)X(X'X)^(-1)c
  cblas_dgemm(
                CblasRowMajor,
                CblasNoTrans,
                CblasNoTrans,
                second_col,
                second_col,
                col,
                1.0,
                cT_XTX_inverse_XT_D_X_XTX_inverse,
                col,
                c,
                second_col,
                0.0,
                cT_XTX_inverse_XT_D_X_XTX_inverse_c,
                second_col
                );

  double ret = cT_XTX_inverse_XT_D_X_XTX_inverse_c[0];

  delete[] cT_XTX_inverse_XT;
  delete[] D_X_XTX_inverse;
  delete[] cT_XTX_inverse_XT_D_X_XTX_inverse;
  delete[] cT_XTX_inverse_XT_D_X_XTX_inverse_c;
  delete[] rows_start;
  delete[] rows_end;
  delete[] col_indx;

  return ret;
}

double compute_var_cTbeta_hat_sparse_triangle(const double* c, const double* X, const double* XTX_inverse_XT, const double* X_XTX_inverse, 
                                              double* D_values, int scan_number, int col, int second_col, int n){
  double* cT_XTX_inverse_XT = new double[second_col * scan_number];
  double* D_X_XTX_inverse = new double[scan_number * col];
  double* cT_XTX_inverse_XT_D_X_XTX_inverse = new double[second_col * col];
  double* cT_XTX_inverse_XT_D_X_XTX_inverse_c = new double[second_col]; 

  // compute c'(X'X)^(-1)X'
  cblas_dgemm(
                CblasRowMajor,
                CblasTrans,
                CblasNoTrans,
                second_col,
                scan_number,
                col,
                1.0,
                c,
                second_col,
                XTX_inverse_XT,
                scan_number,
                0.0,
                cT_XTX_inverse_XT,
                scan_number
                );


  // reference: https://software.intel.com/en-us/mkl-developer-reference-c-mkl-sparse-create-csr
  int* rows_start = new int[scan_number];
  int* rows_end = new int[scan_number];
  int* col_indx = new int[n * n + scan_number - n];
  for(int i = 0; i < n; i++){
    rows_start[i] = i * n;
    rows_end[i] = i * n + n;
    for(int j = 0; j < n; j++){
        col_indx[i*n+j] = j;
    }
  }
  for(int i = n; i < scan_number; i++){
    rows_start[i] = n * n + i - n;
    rows_end[i] = n * n + i - n + 1;
    col_indx[n*n+i-n] = i;
  }

  sparse_matrix_t D;
  matrix_descr descr;
  descr.type = sparse_matrix_type_t::SPARSE_MATRIX_TYPE_SYMMETRIC;
  descr.diag = sparse_diag_type_t::SPARSE_DIAG_NON_UNIT;
  descr.mode = sparse_fill_mode_t::SPARSE_FILL_MODE_LOWER;

  mkl_sparse_d_create_csr(&D, SPARSE_INDEX_BASE_ZERO, scan_number, scan_number, rows_start, rows_end, col_indx, D_values);

  /* 
   * These optimization rules will actually make its performance worse than normal
   *
   */
  // reference: https://software.intel.com/en-us/mkl-developer-reference-c-mkl-sparse-set-mm-hint
  //mkl_sparse_set_mm_hint(D, SPARSE_OPERATION_NON_TRANSPOSE, descr, SPARSE_LAYOUT_ROW_MAJOR, scan_number, 1);

  // reference: https://software.intel.com/en-us/mkl-developer-reference-c-mkl-sparse-set-memory-hint
  //mkl_sparse_set_memory_hint(D, SPARSE_MEMORY_AGGRESSIVE); 

  // reference: https://software.intel.com/en-us/mkl-developer-reference-c-mkl-sparse-optimize
  //mkl_sparse_optimize(D);

  // reference: https://software.intel.com/en-us/mkl-developer-reference-c-mkl-sparse-mm
  // compute D_r^(*)X(X'X)^(-1)
  mkl_sparse_d_mm(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, D, descr, SPARSE_LAYOUT_ROW_MAJOR, X_XTX_inverse, col, col, 0.0, D_X_XTX_inverse, col);

  // reference: https://software.intel.com/en-us/mkl-developer-reference-c-mkl-sparse-destroy
  mkl_sparse_destroy(D);

  // compute c'(X'X)^(-1)X'D_r^(*)X(X'X)^(-1)
  cblas_dgemm(
                CblasRowMajor,
                CblasNoTrans,
                CblasNoTrans,
                second_col,
                col,
                scan_number,
                1.0,
                cT_XTX_inverse_XT,
                scan_number,
                D_X_XTX_inverse,
                col,
                0.0,
                cT_XTX_inverse_XT_D_X_XTX_inverse,
                col
                );

  // compute c'(X'X)^(-1)X'D_r^(*)X(X'X)^(-1)c
  cblas_dgemm(
                CblasRowMajor,
                CblasNoTrans,
                CblasNoTrans,
                second_col,
                second_col,
                col,
                1.0,
                cT_XTX_inverse_XT_D_X_XTX_inverse,
                col,
                c,
                second_col,
                0.0,
                cT_XTX_inverse_XT_D_X_XTX_inverse_c,
                second_col
                );

  double ret = cT_XTX_inverse_XT_D_X_XTX_inverse_c[0];

  delete[] cT_XTX_inverse_XT;
  delete[] D_X_XTX_inverse;
  delete[] cT_XTX_inverse_XT_D_X_XTX_inverse;
  delete[] cT_XTX_inverse_XT_D_X_XTX_inverse_c;
  delete[] rows_start;
  delete[] rows_end;
  delete[] col_indx;

  return ret;
}

bool motion_correction(vector<vector<double>>& motion_data, vector<double>& FD_v, vector<double>& RMS, double threshold, int scan_number, string filename_prefix, string filename_suffix){
    int scan_n = scan_number;
    vector<double> last_motion_data = motion_data[scan_number-1];
    vector<int> num(4, 0);
    assert(scan_number < 10000);
    int index = 3;
    while(scan_number >= 1){
        num[index] = scan_number % 10;
        scan_number /= 10;
        index--;
    }
    string scan = "";
    for(int i = 0; i < 4; i++){
        scan += to_string(num[i]);
    }
    string filename = filename_prefix + scan + filename_suffix;

    cout << "Looking for motion file " << filename  << " ..." << endl;

    ifstream myfile(filename, ios_base::in);
    double in;
    vector<double> current_motion_data(9);
    for(int i = 0; i < 9; i++){
        myfile >> in;
        current_motion_data[i] = in;
    }
    myfile.close();
    motion_data[scan_n] = current_motion_data;

    const double degree_to_radian = M_PI / 180;
    double FD = abs(current_motion_data[4]-last_motion_data[4]) 
              + abs(current_motion_data[5]-last_motion_data[5]) 
              + abs(current_motion_data[6]-last_motion_data[6])
              + abs(current_motion_data[1]-last_motion_data[1]) * 50 * degree_to_radian;
              + abs(current_motion_data[2]-last_motion_data[2]) * 50 * degree_to_radian;
              + abs(current_motion_data[3]-last_motion_data[3]) * 50 * degree_to_radian;

    FD_v[scan_n] = FD;
    RMS[scan_n] = current_motion_data[8];
    return FD < threshold;
}



