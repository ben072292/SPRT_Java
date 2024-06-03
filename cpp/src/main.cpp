
/*
 * This code is customized by Weicong and Curtis to show a module to
 * handle the weight lifting part and computes the
 * formula 3 in paper "Dynamic adjustment of stimuli in real time functional
 * magnetic resonance imaging"
 */
#include "numerical.h"
#include <iomanip>  // std::setw
#include <unistd.h> // usleep()
using namespace std;

typedef cilk::reducer<cilk::op_add<int>> Int_Reducer;

extern int SCAN_SIZE;
extern int DIMENSION_SIZE;

int main(int argc, char *argv[])
{
  if (argc != 3)
  {
    cout << "Usage: " << argv[0] << " Scan_Size Dimension_Size" << endl;
    exit(1);
  }
  SCAN_SIZE = atoi(argv[1]);
  DIMENSION_SIZE = atoi(argv[2]);

  int row, col, K_blk;
  double Z; // Z scores to use
  int temp; // store each scan number to compute Z
  row = SCAN_SIZE;
  col = DIMENSION_SIZE;
  /* the value of the second column is always 1 because this is the response value for each voxel */
  int second_col = 1;

  int num_of_C;
  double **C; // C is a selector and actually has the same length with B
  double theta_0, theta_1;
  bool use_pre_assigned_theta1 = false;
  double alpha, beta; // alpha and beta are the stoping rule's parameters
  double crossed_easy_percent, crossed_hard_percent;
  char flag;
  char load_default;
  double motion_threshold;

  cout << "Load default setting (y/N)? " << endl;
  cin >> load_default;
  if (toupper(load_default) != 'Y')
  {
    cout << "Please input parameter K_block: (enter 0 if you wish to set static \u03D1 1)";
    cin >> K_blk;
    cout << "Please input parameter \u03D1 0:";
    cin >> theta_0;
    if (K_blk == 0)
    {
      use_pre_assigned_theta1 = true;
      cout << "Please input parameter \u03D1 1:";
      cin >> theta_1;
    }

    cout << "Please input how many selectors are needed ";
    cin >> num_of_C;
    if (num_of_C < 2)
    {
      cout << "Warning: You can not have less than 2 selectors, otherwise the program will crash. Default set to 2 selctors." << endl;
      num_of_C = 2;
    }
    C = new double *[num_of_C];

    for (int i = 0; i < num_of_C; i++)
    {
      C[i] = new double[col];
      cout << "Please input C" << i + 1 << " (please note that C1 has the same length of B):";
      for (int j = 0; j < col; j++)
        cin >> C[i][j];
    }

    cout << "Please input alpha and beta values:";
    cin >> alpha;
    cin >> beta;

    cout << "Please input Z score:";
    cin >> Z;

    cout << "Please input percentage of crossed_easy and crossed_hard: ";
    cin >> crossed_easy_percent;
    cin >> crossed_hard_percent;

    cout << "Please input motion threshold: ";
    cin >> motion_threshold;
  }
  else
  { // below is the default setting for quick testing
    num_of_C = 4;
    K_blk = 78;
    theta_0 = 0;
    alpha = 0.001;
    beta = 0.1;
    Z = 3.12;
    crossed_easy_percent = 0.6;
    crossed_hard_percent = 0.6;
    C = new double *[num_of_C];
    C[0] = new double[col]{0, 1, 0, 0, 0, 0, 0, 0};
    C[1] = new double[col]{0, 0, 1, 0, 0, 0, 0, 0};
    C[2] = new double[col]{0, 1, -1, 0, 0, 0, 0, 0};
    C[3] = new double[col]{0, -1, 1, 0, 0, 0, 0, 0};
    motion_threshold = 0.9;
  }

  /* double check the parameters */
  cout << "So the parameters are: " << endl;
  cout << "    " << "K_block is " << K_blk << endl;
  cout << "    " << "\u03D1 0 is " << theta_0 << endl;
  // cout << "    " << "\u03D1 1 is " << theta_1 << endl;
  cout << "    " << "alpha is " << alpha << endl;
  cout << "    " << "beta is " << beta << endl;
  cout << "    " << "Z score is " << Z << endl;
  cout << "    " << "crossed_easy_percent: " << crossed_easy_percent << "%" << endl;
  cout << "    " << "crossed_hard_percent: " << crossed_hard_percent << "%" << endl;
  crossed_easy_percent /= 100.0; // percentage to decimal
  crossed_hard_percent /= 100.0;
  cout << "    " << "motion threshold: " << motion_threshold << endl;

  for (int i = 0; i < num_of_C; i++)
  {
    cout << "    " << "C" << i + 1 << " is [";
    for (int j = 0; j < col; j++)
      cout << C[i][j] << " ";
    cout << "]" << endl;
  }

  cout << "Please enter y/n to continue or redo the input: ";
  cin >> flag;
  if (toupper(flag) != 'Y')
  {
    exit(1);
  }

  vector<double> boundary = stop_boundary(alpha, beta);
  // cout << boundary[0] << endl << boundary[1] << endl;
  if (boundary.size() != 2)
  {
    throw "Boundaries for stop rules are not correct.";
  }

  string file_path_x =
      "./Latest_data/design_easy.txt";

  /* resonse matrix needs to be computed iteratively reading docs */
  string file_path_y =
      "./Latest_data/";

  vector<vector<double>> response;
  vector<int> dimensions;

  /* create cout ostream reducer to output results in order */
  cilk::reducer<cilk::op_ostream> cout_r(cout);

  ofstream myfile1;
  ofstream myfile2;
  ofstream myfile3;                             // used to output _beta_hat value for testing
  myfile1.open("SPRT_statistics.csv");          // stores computation and activation summary
  myfile2.open(file_path_y + "Activation.csv"); // stores activation details used for trigger pulse
  myfile2 << "Scan number,";
  for (int i = 1; i <= num_of_C; i++)
    myfile2 << "Contrast " << i << " Easy, Contrast" << i << " Hard,";
  myfile2 << endl;

  myfile1 << "alpha: " << alpha << " beta: " << beta << " lower bound: " << min(boundary[0], boundary[1]) << " upper bound: " << max(boundary[0], boundary[1]) << ",";
  for (int i = 0; i < num_of_C; i++)
  {
    for (int j = 0; j < col; j++)
    {
      myfile1 << C[i][j];
    }
    myfile1 << ",,";
  }
  myfile1 << endl;
  myfile1 << "Scan number,";
  for (int i = 1; i <= num_of_C; i++)
  {
    myfile1 << "C" << i << " cross upper, C" << i << " cross lower,";
  }
  myfile1 << "Speed (sec)" << endl;

  /* Creat folder to store test output */
  mkdir("test_files", S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
  for (int i = 1; i <= row; i++)
    mkdir(("test_files/" + to_string(i)).c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);

  /*
   * we would like to collect the first scan here
   * in order to retrive voxel size information
   * This information is used to determine the ROI region
   * knowing the ROI region can boost SPRT's analysis speed
   *
   * HERE ASSUME ROI REMAINS THE SAME ACROSS ALL SCANS
   */

  cout << "Waiting for the first scan to arrive" << endl;
  // Wait for second scan arrived to avoid synchronization issue. (first scan is bold0.txt)
  while (!check_file_existence(file_path_y + "bold1.txt"))
  {
    // busy wait
  }
  cout << "Caught the first scan, construct data structure..." << endl;

  /* now we have got dimension information stored in dimensions variable. */
  read_scan(response, file_path_y, dimensions, 0);

  // Now we have enough information to construct theta1 vector
  vector<vector<double>> theta_1_C(num_of_C, vector<double>(dimensions[0] * dimensions[1] * dimensions[2], 0.0));

  /* now we have got "useful" voxel number which is been within ROI */
  int ROI_voxel_number = 0; // all masked voxels
  vector<bool> is_within_ROI(response[0].size(), false);
  for (int i = 0; i < response[0].size(); i++)
  {
    if (response[0][i] != 0.000000)
    {
      ROI_voxel_number++;
      is_within_ROI[i] = true;
    }
  }
  cout << "ROI map constructed! Total ROI voxel number: " << ROI_voxel_number << endl;

  double *X;

  vector<vector<double>> motion_data(row, vector<double>(9, 0.0));
  vector<double> FD(row, 0.0);
  vector<double> RMS(row, 0.0);
  vector<bool> motion_correction_result(row, true);

  /* Offline test code for motion files
  ofstream myfile4;
  for(int i = 10; i < row; i++){
    motion_correction(motion_data, FD, RMS, motion_threshold, i, "./nifti/Dump-", ".txt");
  }

  cout << "Output Global Test Files ..." << endl;
  myfile4.open("motion.csv");
  myfile4 << "Threshold: " << to_string(motion_threshold) << endl;
  myfile4 << "Scan Number:,";
  for(int i = 0; i < row; i++){
    myfile4 << "Scan " + to_string(i+1) << ",";
  }
  myfile4 << endl;
  for(int i = 0; i < motion_data[0].size(); i++){
    myfile4 << "Column " << i+1 << ",";
    for(int j = 0; j < row; j++){
      myfile4 << motion_data[j][i] << ",";
    }
    myfile4 << endl;
  }
  myfile4 << "FD:,";
  for(int i = 0; i <row; i++){
    myfile4 << FD[i] << ",";
  }
  myfile4 << endl;
  myfile4 << "RMS:,";
  for(int i = 0; i <row; i++){
    myfile4 << RMS[i] << ",";
  }
  myfile4 << endl;
  myfile4 << "Result:,";
  for(int i = 0; i < row; i++){
    if(i < K_blk){
      myfile4 << "Pass,";
      continue;
    }
    if(motion_correction_result[i]) myfile4 << "Good,";
    else myfile4 << "Skip,";
  }
  myfile4 << endl;
  myfile4.close();
  */

  if (!use_pre_assigned_theta1)
  {
    X = new double[K_blk * col];
    init_mat(X, K_blk, col, file_path_x);
    /*
     * Here we we collect first K_blk blocks of scans to dynamically
     * Compute theta_1
     */
    for (int scan_number = 2; scan_number <= K_blk; scan_number++)
    { // since we have collected the first scan, we start from the second scan
      /* Check existence of file */
      cout << "Waiting for bold" + to_string(scan_number - 1) + ".txt..." << endl;
      if (scan_number != row)
      {
        while (!check_file_existence(file_path_y + "bold" + to_string(scan_number) + ".txt"))
        {
          // busy wait here
        }
      }
      else
      {
        while (!check_file_existence(file_path_y + "bold" + to_string(scan_number - 1) + ".txt"))
        {
          /* busy wait here */
        }
        usleep(1000); // wait 1000 milli seconds
      }
      cout << "File arrived. start reading in..." << endl;

      if (scan_number == K_blk)
        motion_correction(motion_data, FD, RMS, motion_threshold, scan_number - 1, "./nifti/Dump-", ".txt");
      /* 2-D array response holds the response values for all data points for all time slots */
      read_scan(response, file_path_y, dimensions, scan_number - 1);
    }

    cout << "Finish Collecting first " << K_blk << " Scans." << endl;

    cout << "Estimating \u03D11 for each contrast ... ";

    estimate_theta1(theta_1_C, response, dimensions, X, C, num_of_C, K_blk, col, second_col, Z, is_within_ROI);

    // output theta1 value
    for (int i = 0; i < theta_1_C.size(); i++)
    {
      write_out_voxel_level_info_to_file("./test_files/" + to_string(K_blk) + "/theta_1 with C " + to_string(i) + ".txt",
                                         theta_1_C[i],
                                         dimensions, true);
    }

    cout << "Complete!" << endl;
    usleep(500);
    cout << "Finished Computing All theta_1s At Voxel level." << endl
         << "Start SPRT Analysis Using the computed theta_1s." << endl;
    delete[] X;
  }
  else
  { // If we use pre_assigned theta_1 value
    vector<double> v(dimensions[0] * dimensions[1] * dimensions[2]);
    fill(v.begin(), v.end(), theta_1);
    for (int i = 0; i < num_of_C; i++)
    {
      theta_1_C.push_back(v);
    }
  }

  double ***alternative_beta = new double **[row];
  for (int i = 0; i < row; i++)
  {
    alternative_beta[i] = new double *[num_of_C];
    for (int j = 0; j < num_of_C; j++)
    {
      alternative_beta[i][j] = new double[dimensions[0] * dimensions[1] * dimensions[2]]();
    }
  }

  for (int i = 0; i < num_of_C; i++)
  {
    for (int j = 0; j < dimensions[0] * dimensions[1] * dimensions[2]; j++)
    {
      alternative_beta[K_blk - 1][i][j] = theta_1_C[i][j];
    }
  }

  /* two boolean arrays to indicate if this scan has cross upper or lower bounds for each C respectively*/
  bool cross_upper[num_of_C][dimensions[0] * dimensions[1] * dimensions[2]];
  bool cross_lower[num_of_C][dimensions[0] * dimensions[1] * dimensions[2]];
  /* set to false as default */
  for (int i = 0; i < num_of_C; i++)
  {
    fill_n(cross_upper[i], dimensions[0] * dimensions[1] * dimensions[2], false);
    fill_n(cross_lower[i], dimensions[0] * dimensions[1] * dimensions[2], false);
  }

  int **scan_number_for_result_lower = new int *[num_of_C]; // stores the scan number this voxel first cross lower bound
  int **scan_number_for_result_upper = new int *[num_of_C]; // stores the scan number this voxel first cross upper bound
  /* initialize */
  for (int i = 0; i < num_of_C; i++)
  {
    scan_number_for_result_lower[i] = new int[dimensions[0] * dimensions[1] * dimensions[2]];
    fill_n(scan_number_for_result_lower[i], dimensions[0] * dimensions[1] * dimensions[2], 0);
    scan_number_for_result_upper[i] = new int[dimensions[0] * dimensions[1] * dimensions[2]];
    fill_n(scan_number_for_result_upper[i], dimensions[0] * dimensions[1] * dimensions[2], 0);
  }

  /* initialize opadd reducer counters */
  Int_Reducer cross_upper_bound_voxel_counter[num_of_C], cross_lower_bound_voxel_counter[num_of_C];

  bool crossed_easy = false;   // whether crossed easy level bound
  bool swapped_matrix = false; // whether switched matrix yet

  /* Declare and intialize design matrix */
  X = new double[row * col];
  init_mat(X, row, col, file_path_x);
  double *XTX_inverse = new double[col * col]; // will be used to compute beta_hat

  chrono::high_resolution_clock::time_point t1, t2; // for store current time
  chrono::duration<double> t;                       // for counting time difference

  /*
   * initialize var(c'beta_hat)
   */
  double ***var_cT_beta_hat = new double **[row];
  for (int i = 0; i < row; i++)
  {
    var_cT_beta_hat[i] = new double *[num_of_C];
    for (int j = 0; j < num_of_C; j++)
      var_cT_beta_hat[i][j] = new double[dimensions[0] * dimensions[1] * dimensions[2]]();
  }

  /* ********************************************************************************************************** */
  // store all beta_hat values, a scan_number * col * (dimension[0] * dimensions[1] * dimensions[2]) matrix
  double ***_beta_hat = new double **[row];
  for (int i = 0; i < row; i++)
  {
    _beta_hat[i] = new double *[col];
    for (int j = 0; j < col; j++)
      _beta_hat[i][j] = new double[dimensions[0] * dimensions[1] * dimensions[2]]();
  }

  double ***Z_score = new double **[row];
  for (int i = 0; i < row; i++)
  {
    Z_score[i] = new double *[num_of_C];
    for (int j = 0; j < num_of_C; j++)
      Z_score[i][j] = new double[dimensions[0] * dimensions[1] * dimensions[2]]();
  }

  double ***SPRT = new double **[row];
  for (int i = 0; i < row; i++)
  {
    SPRT[i] = new double *[num_of_C];
    for (int j = 0; j < num_of_C; j++)
      SPRT[i][j] = new double[dimensions[0] * dimensions[1] * dimensions[2]]();
  }

  /* ********************************************************************************************************** */

  /* start SPRT anlysis */
  for (int scan_number = K_blk + 1; scan_number <= row; scan_number++)
  {

    // motion larger than threshold will be discarded.
    motion_correction_result[scan_number - 1] = motion_correction(motion_data, FD, RMS, motion_threshold, scan_number - 1, "./nifti/Dump-", ".txt");

    compute_XTX_inverse(XTX_inverse, X, scan_number, col); // compute XTX_inverse for only once for each loop, here we truncate X matrix to scan_numer rows

    /* initialized oppadd reducer counters to 0 */
    for (int i = 0; i < num_of_C; i++)
    {
      cross_upper_bound_voxel_counter[i].set_value(0);
      cross_lower_bound_voxel_counter[i].set_value(0);
    }

    /*
     * Initialize variables for computing var(c'beta_hat)
     */
    double *XTX_inverse_XT = new double[col * scan_number](); // 8 * 238
    double *X_XTX_inverse = new double[scan_number * col]();  // 238 * 8
    compute_XTX_inverse_XT(XTX_inverse_XT, X, XTX_inverse, scan_number, col);
    compute_X_XTX_inverse(X_XTX_inverse, X, XTX_inverse, scan_number, col);
    vector<double> H_diagnal;
    // only once per scan
    // no need to clear, function handles
    compute_H_diagnal(H_diagnal, X_XTX_inverse, X, scan_number, col);

    /* Check existence of file */
    cout << "Waiting for bold" + to_string(scan_number - 1) + ".txt..." << endl;
    if (scan_number != row)
    {
      while (!check_file_existence(file_path_y + "bold" + to_string(scan_number) + ".txt"))
      {
        // busy wait here
      }
    }
    else
    {
      while (!check_file_existence(file_path_y + "bold" + to_string(scan_number - 1) + ".txt"))
      {
        /* busy wait here */
      }
      usleep(100); // wait 1000 milli seconds
    }
    cout << "File arrived. start reading in..." << endl;

    /* 2-D array response holds the response values for all data points for all time slots */
    read_scan(response, file_path_y, dimensions, scan_number - 1);
    cout << "Read complete. " << scan_number << " scans have arrived. Starting SPRT analysis for them..." << endl;

    estimate_theta1(theta_1_C, response, dimensions, X, C, num_of_C, scan_number, col, second_col, Z, is_within_ROI);

    for (int i = 0; i < num_of_C; i++)
    {
      for (int j = 0; j < dimensions[0] * dimensions[1] * dimensions[2]; j++)
      {
        alternative_beta[scan_number - 1][i][j] = theta_1_C[i][j];
      }
    }

    /* start counting time */
    t1 = chrono::high_resolution_clock::now();

    /* for each point in X-Y-Z space, compute the response array Y */
    cilk_for(int x = 0; x < dimensions[0]; x++)
    {
      for (int y = 0; y < dimensions[1]; y++)
      {
        for (int z = 0; z < dimensions[2]; z++)
        {
          if (is_within_ROI[x * dimensions[1] * dimensions[2] + y * dimensions[2] + z])
          { // only prepare data and run SPRT when voxel is within ROI region
            int pos = x * dimensions[1] * dimensions[2] + y * dimensions[2] + z;
            double *Y = new double[scan_number * second_col]; // Initialize the response matrix
            double *beta_hat = new double[col * second_col];  // DIMENSION_SIZE * 1

            // fix X - Y - Z axis value and fill in the response for
            // response array Y
            for (int temp = 0; temp < scan_number; temp++)
            {
              Y[temp] = response[temp][pos];
            }

            compute_beta_hat(X, Y, beta_hat, XTX_inverse, scan_number, col, second_col);

            /* ****************** */
            for (int i = 0; i < col; i++)
              _beta_hat[scan_number - 1][i][pos] = beta_hat[i];
            /* ****************** */

            /*
             * start computing var(c'beta_hat)
             */
            double *R = new double[scan_number];
            compute_R(R, Y, X, beta_hat, scan_number, col, second_col);
            double *D_values = new double[scan_number];
            generate_D_values(D_values, R, H_diagnal, scan_number);
            for (int i = 0; i < num_of_C; i++)
              var_cT_beta_hat[scan_number - 1][i][pos] = compute_var_cTbeta_hat_sparse(C[i], X, XTX_inverse_XT, X_XTX_inverse,
                                                                                       D_values, scan_number, col, second_col);

            /*
             * For testing purpose, we compute and store Z score
             */
            compute_Z_score(Z_score[scan_number - 1], C, beta_hat, var_cT_beta_hat[scan_number - 1], num_of_C, pos, col, second_col);

            double sprt[num_of_C]; // must be declared insde the loop for parallel programming
            for (int i = 0; i < num_of_C; i++)
            {
              /* SPRT computation */
              sprt[i] = compute_SPRT(beta_hat, col, C[i], theta_0, theta_1_C[i][pos], var_cT_beta_hat[scan_number - 1][i][pos]);
              SPRT[scan_number - 1][i][pos] = sprt[i]; // For testing purpose

              /* statistical processing */
              if (sprt[i] > max(boundary[0], boundary[1]))
              {
                (*cross_upper_bound_voxel_counter[i])++;
                if (!cross_upper[i][pos])
                {
                  scan_number_for_result_upper[i][pos] = scan_number;
                  cross_upper[i][pos] = true;
                }
              }
              if (sprt[i] < min(boundary[0], boundary[1]))
              {
                (*cross_lower_bound_voxel_counter[i])++;
                if (!cross_lower[i][pos])
                {
                  scan_number_for_result_lower[i][pos] = scan_number;
                  cross_lower[i][pos] = true;
                }
              }
            }
            delete[] R;
            delete[] Y;
            delete[] beta_hat;
            delete[] D_values;
          } // end is_with_ROI
        } // end dimensions[2]
      } // end dimensions[1]
    } // end of cilk for loop dimensions[0]

    /* output to stdout */
    *cout_r << endl
            << endl
            << "---------------------- Round " << scan_number << ": -----------------------" << endl;
    *cout_r << "   " << "Lower boundary is " << min(boundary[0], boundary[1]) << ". Upper boundary is " << max(boundary[0], boundary[1]) << endl;
    for (int i = 0; i < num_of_C; i++)
    {
      *cout_r << "            " << (double)cross_upper_bound_voxel_counter[i].get_value() << " (" << 100 * (double)cross_upper_bound_voxel_counter[i].get_value() / (double)(ROI_voxel_number) << "%) crossed upperbound for C" << i + 1 << endl;
      *cout_r << "            " << (double)cross_lower_bound_voxel_counter[i].get_value() << " (" << 100 * (double)cross_lower_bound_voxel_counter[i].get_value() / (double)(ROI_voxel_number) << "%) crossed lowerbound for C" << i + 1 << endl;
    }

    myfile1 << scan_number << ",";
    for (int i = 0; i < num_of_C; i++)
    {
      myfile1 << (double)cross_upper_bound_voxel_counter[i].get_value() << "," << (double)cross_lower_bound_voxel_counter[i].get_value() << ",";
    }

    /* Stop ticking and ouput timer result */
    t2 = chrono::high_resolution_clock::now();
    t = t2 - t1;
    cout << "            Processed in " << t.count() << " seconds" << endl;
    cout << "---------------------------------------------------------" << endl
         << endl;

    myfile1 << t.count() << endl;

    /*
     * Output into file for trigger pulse and then do cleanup
     * Format: scan_number c_matrix_number activation_code
     * scan_number: number of scans
     * c_matrix_number: either C1 or C2 or C3
     * activation code: 0 - cross lower bound
     *                  1 - cross upper bound
     *                  2 - cross both bounds in a single scan
     *                 -1 - within bounds
     */
    if (scan_number > K_blk)
    {
      myfile2 << scan_number << ",";
      for (int i = 0; i < num_of_C; i++)
      {
        if (((double)cross_upper_bound_voxel_counter[i].get_value() + (double)cross_lower_bound_voxel_counter[i].get_value()) / (double)(ROI_voxel_number) > crossed_easy_percent)
          myfile2 << "Stop,";
        else
          myfile2 << "Continue,";

        if (((double)cross_upper_bound_voxel_counter[1].get_value() + (double)cross_lower_bound_voxel_counter[1].get_value()) / (double)(ROI_voxel_number) > crossed_hard_percent)
          myfile2 << "Stop,";
        else
          myfile2 << "Continue,";
      }
      myfile2 << endl;

      /* If we have not swiched design matrix, wait for the file
       * indicating which scan it switch the difficulty level,
       * then swap the design matrix accordingly
       * Swapping matrix runs only once
       */
      if (crossed_easy && !swapped_matrix)
      {
        cout << "Detects crossed bound(s)." << endl;
        usleep(100); // sleep 100ms for files to transfer back from lumia box controller.
        if (check_file_existence("./Latest_data/found_activation_stopeasy.txt"))
        {
          ifstream in("./Latest_data/found_activation_stopeasy.txt");
          int n;
          in >> n;
          in.close();
          assemble_new_degign_matrix("./Latest_data/design_easy.txt", "./Latest_data/design_hard.txt", "./Latest_data/design_new.txt", row, col, n);
          init_mat(X, row, col, "./Latest_data/design_new.txt");
          swapped_matrix = true;
          cout << "Swap design matrix at " << n << " scans" << endl;
          myfile2 << "Swap design matrix at " << n << " scans" << endl;
        }
      }
    }

    /* ********************************************************************************************************** */
    /* start counting time */
    // cout << "Generating test files ..." << endl;
    // t1 = chrono::high_resolution_clock::now();
    // /* output beta_hat of each scan */
    // for(int i=0; i<col; i++){
    //   myfile3.open("./test_files/" + to_string(scan_number) + "/beta_hat at regressor " + to_string(i) + ".txt");
    //   for(int x = 0; x < dimensions[0]; x++){
    //     myfile3 << "Slice " << x << ": " << endl;
    //     for(int y = 0; y < dimensions[1]; y++){
    //       for(int z = 0; z < dimensions[2]; z++){
    //         myfile3 << _beta_hat[scan_number-1][i][x * dimensions[1] * dimensions[2] + y * dimensions[2] + z] << " ";
    //       }
    //       myfile3 << endl;
    //     }
    //     myfile3 << endl << endl;
    //   }
    //   myfile3.close();
    // }

    // /* output Z_score of each scan */
    // for(int i = 0; i < num_of_C; i++){
    //   myfile3.open("./test_files/" + to_string(scan_number) + "/Z score at contrast " + to_string(i+1) + ".txt");
    //   for(int x = 0; x < dimensions[0]; x++){
    //     for(int y = 0; y < dimensions[1]; y++){
    //       for(int z = 0; z < dimensions[2]; z++){
    //         myfile3 << Z_score[scan_number-1][i][x * dimensions[1] * dimensions[2] + y * dimensions[2] + z] << " ";
    //       }
    //       myfile3 << endl;
    //     }
    //     myfile3 << endl << endl;
    //   }
    //   myfile3.close();
    // }

    // /* output sandwich covariance matrix of each scan */
    // for(int i = 0; i < num_of_C; i++){
    //   myfile3.open("./test_files/" + to_string(scan_number) + "/covariance matrix at contrast " + to_string(i+1) + ".txt");
    //   for(int x = 0; x < dimensions[0]; x++){
    //     for(int y = 0; y < dimensions[1]; y++){
    //       for(int z = 0; z < dimensions[2]; z++){
    //         myfile3 << var_cT_beta_hat[scan_number-1][i][x * dimensions[1] * dimensions[2] + y * dimensions[2] + z] << " ";
    //       }
    //       myfile3 << endl;
    //     }
    //     myfile3 << endl << endl;
    //   }
    //   myfile3.close();
    // }
    // /* ********************************************************************************************************** */

    // delete[] X_XTX_inverse;
    // delete[] XTX_inverse_XT;
    // /* Stop ticking and ouput timer result */
    // t2 = chrono::high_resolution_clock::now();
    // t = t2 - t1;
    // cout << "Processed in " << t.count() << " seconds" << endl << endl << endl;
  } // end of SPRT computation

  /* Close SPRT process file */
  myfile1.close();
  myfile2.close();

  /* ********************************************************************************************************** */

  // cout << "Output Global Test Files ..." << endl;
  // myfile1.open("motion.csv");
  // myfile1 << "Threshold: " << to_string(motion_threshold) << endl;
  // myfile1 << "Scan Number:,";
  // for(int i = 0; i < row; i++){
  //   myfile1 << "Scan " + to_string(i+1) << ",";
  // }
  // myfile1 << endl;
  // for(int i = 0; i < motion_data[0].size(); i++){
  //   myfile1 << "Column " << i+1 << ",";
  //   for(int j = 0; j < row; j++){
  //     myfile1 << motion_data[j][i] << ",";
  //   }
  //   myfile1 << endl;
  // }
  // myfile1 << "FD:,";
  // for(int i = 0; i <row; i++){
  //   myfile1 << FD[i] << ",";
  // }
  // myfile1 << endl;
  // myfile1 << "RMS:,";
  // for(int i = 0; i <row; i++){
  //   myfile1 << RMS[i] << ",";
  // }
  // myfile1 << endl;
  // myfile1 << "Result:,";
  // for(int i = 0; i < row; i++){
  //   if(i < K_blk){
  //     myfile1 << "Pass,";
  //     continue;
  //   }
  //   if(motion_correction_result[i]) myfile1 << "Good,";
  //   else myfile1 << "Skip,";
  // }
  // myfile1 << endl;
  // myfile1.close();

  // /*
  //  * Output Voxel Level Info: Activation
  //  * of first scan on over upper/lower bound
  //  */
  // for(int i=0; i<num_of_C; i++){
  //   myfile1.open("Voxel Level Info for contrast " + to_string(i+1) + ".csv");
  //   myfile1 << "Voxel Number,First Scan Cross Lower,First Scan Cross Upper" << endl;
  //   for (int x = 0; x < dimensions[0]; x++) {
  //     for (int y = 0; y < dimensions[1]; y++) {
  //       for (int z = 0; z < dimensions[2]; z++) {
  //         if(scan_number_for_result_upper[i][x * dimensions[1] * dimensions[2] + y * dimensions[2] + z] != 0
  //         || scan_number_for_result_lower[i][x * dimensions[1] * dimensions[2] + y * dimensions[2] + z] != 0)
  //           myfile1 <<"[" << setw(3)
  //                   << x
  //                   << setw(3)
  //                   << y
  //                   << setw(3)
  //                   << z
  //                   << "],"
  //                   << scan_number_for_result_lower[i][x * dimensions[1] * dimensions[2] + y * dimensions[2] + z]
  //                   << ","
  //                   << scan_number_for_result_upper[i][x * dimensions[1] * dimensions[2] + y * dimensions[2] + z]
  //                   << endl;
  //       }
  //     }
  //   }
  //   myfile1.close();
  // }

  // /*
  //  * Output SPRT time series data
  //  */
  // for(int i = 0; i < num_of_C; i++){
  //   myfile1.open("SPRT time series for contrast " + to_string(i+1) + ".csv");
  //   for (int x = 0; x < dimensions[0]; x++) {
  //     for (int y = 0; y < dimensions[1]; y++) {
  //       for (int z = 0; z < dimensions[2]; z++) {
  //         myfile1 <<"[" << setw(3)
  //                   << x
  //                   << setw(3)
  //                   << y
  //                   << setw(3)
  //                   << z
  //                   << "],";
  //         for(int n = K_blk; n < row; n++){
  //           myfile1 << SPRT[n][i][x * dimensions[1] * dimensions[2] + y * dimensions[2] + z] << ",";
  //         }
  //         myfile1 << endl;
  //       }
  //     }
  //   }
  //   myfile1.close();
  // }

  cout << "Do you want to output time series data for specific voxels?"
       << "\nIf yes, please put voxel coordinates into a file named"
       << "\n\"voxels_list.txt\" separated by line, the format for each voxel is"
       << "\nX Y Z" << endl;
  char output_specific_voxles;
  cin >> output_specific_voxles;
  if (toupper(output_specific_voxles) == 'Y')
  {
    int coor;
    ifstream myfile("voxels_list.txt");
    vector<int> voxels_list;
    while (myfile >> coor)
    {
      voxels_list.push_back(coor);
    }
    if (voxels_list.size() % 3 == 0)
    {
      for (int i = 0; i < num_of_C; i++)
      {
        myfile1.open("Specific_voxel_Info for contrast " + to_string(i + 1) + ".csv");
        myfile1 << "This is specific voxel information for contrast" << to_string(i + 1) << endl
                << endl;
        myfile1 << "Set Z score:," << Z << endl;
        myfile1 << "SPRT set upper bound:," << max(boundary[0], boundary[1]) << endl;
        myfile1 << "SPRT set lower bound:," << min(boundary[0], boundary[1]) << endl;
        myfile1 << "SPRT Value Time Series Data" << endl;
        myfile1 << "X,Y,Z,";
        for (int scan = K_blk + 1; scan <= row; scan++)
          myfile1 << "Scan" << scan << ",";
        myfile1 << endl;
        for (int in = 0; in < voxels_list.size() / 3; in++)
        {
          int pos = voxels_list[in * 3] * dimensions[1] * dimensions[2] + voxels_list[in * 3 + 1] * dimensions[2] + voxels_list[in * 3 + 2];
          myfile1 << voxels_list[in * 3] << "," << voxels_list[in * 3 + 1] << "," << voxels_list[in * 3 + 2] << ",";
          for (int n = K_blk; n < row; n++)
          {
            myfile1 << SPRT[n][i][pos] << ",";
          }
          myfile1 << endl;
        }
        myfile1 << endl;

        myfile1 << "Z Score Time Series Data" << endl;
        myfile1 << "X,Y,Z,";
        for (int scan = K_blk + 1; scan <= row; scan++)
          myfile1 << "Scan" << scan << ",";
        myfile1 << endl;
        for (int in = 0; in < voxels_list.size() / 3; in++)
        {
          int pos = voxels_list[in * 3] * dimensions[1] * dimensions[2] + voxels_list[in * 3 + 1] * dimensions[2] + voxels_list[in * 3 + 2];
          myfile1 << voxels_list[in * 3] << "," << voxels_list[in * 3 + 1] << "," << voxels_list[in * 3 + 2] << ",";
          for (int n = K_blk; n < row; n++)
          {
            myfile1 << Z_score[n][i][pos] << ",";
          }
          myfile1 << endl;
        }
        myfile1 << endl
                << endl
                << endl
                << endl;

        myfile1 << "Set Z score:," << Z << endl;
        myfile1 << "SPRT set upper bound:," << max(boundary[0], boundary[1]) << endl;
        myfile1 << "SPRT set lower bound:," << min(boundary[0], boundary[1]) << endl;
        myfile1 << "SPRT Value Time Series Data" << endl;
        myfile1 << "Note: * after value means over set SPRT upper bound, * before value means below set SPRT lower bound." << endl;
        myfile1 << "X,Y,Z,";
        for (int scan = K_blk + 1; scan <= row; scan++)
          myfile1 << "Scan" << scan << ",";
        myfile1 << endl;
        for (int in = 0; in < voxels_list.size() / 3; in++)
        {
          int pos = voxels_list[in * 3] * dimensions[1] * dimensions[2] + voxels_list[in * 3 + 1] * dimensions[2] + voxels_list[in * 3 + 2];
          myfile1 << voxels_list[in * 3] << "," << voxels_list[in * 3 + 1] << "," << voxels_list[in * 3 + 2] << ",";
          for (int n = K_blk; n < row; n++)
          {
            if (SPRT[n][i][pos] > max(boundary[0], boundary[1]))
              myfile1 << SPRT[n][i][pos] << " *" << ",";
            else if (SPRT[n][i][pos] < min(boundary[0], boundary[1]))
              myfile1 << "* " << SPRT[n][i][pos] << ",";
            else
              myfile1 << SPRT[n][i][pos] << ",";
          }
          myfile1 << endl;
        }
        myfile1 << endl;

        myfile1 << "Z Score Time Series Data" << endl;
        myfile1 << "Note: * after value means over set Z score." << endl;
        myfile1 << "X,Y,Z,";
        for (int scan = K_blk + 1; scan <= row; scan++)
          myfile1 << "Scan" << scan << ",";
        myfile1 << endl;
        for (int in = 0; in < voxels_list.size() / 3; in++)
        {
          int pos = voxels_list[in * 3] * dimensions[1] * dimensions[2] + voxels_list[in * 3 + 1] * dimensions[2] + voxels_list[in * 3 + 2];
          myfile1 << voxels_list[in * 3] << "," << voxels_list[in * 3 + 1] << "," << voxels_list[in * 3 + 2] << ",";
          for (int n = K_blk; n < row; n++)
          {
            if (Z_score[n][i][pos] > Z)
              myfile1 << Z_score[n][i][pos] << " *" << ",";
            else
              myfile1 << Z_score[n][i][pos] << ",";
          }
          myfile1 << endl;
        }
        myfile1 << endl;

        myfile1 << "Theta Time Series Data" << endl;
        myfile1 << "X,Y,Z,";
        for (int scan = K_blk + 1; scan <= row; scan++)
          myfile1 << "Scan" << scan << ",";
        myfile1 << endl;
        for (int in = 0; in < voxels_list.size() / 3; in++)
        {
          int pos = voxels_list[in * 3] * dimensions[1] * dimensions[2] + voxels_list[in * 3 + 1] * dimensions[2] + voxels_list[in * 3 + 2];
          myfile1 << voxels_list[in * 3] << "," << voxels_list[in * 3 + 1] << "," << voxels_list[in * 3 + 2] << ",";
          for (int n = K_blk; n < row; n++)
          {
            myfile1 << alternative_beta[n][i][pos] << ",";
          }
          myfile1 << endl;
        }

        vector<double> lower(row, 0.0);
        vector<double> upper(row, 0.0);
        myfile1 << "SPRT 95% confidence intervals" << endl
                << "In Result section, -1 means less than left 2.5% CI, 1 means greater than right 2.5% CI, 0 means in between (Undecided)" << endl;
        myfile1 << "Label,X,Y,Z,";
        for (int scan = K_blk + 1; scan <= row; scan++)
          myfile1 << "Scan" << scan << ",";
        myfile1 << endl;
        for (int in = 0; in < voxels_list.size() / 3; in++)
        {
          int pos = voxels_list[in * 3] * dimensions[1] * dimensions[2] + voxels_list[in * 3 + 1] * dimensions[2] + voxels_list[in * 3 + 2];
          myfile1 << "Lower," << voxels_list[in * 3] << "," << voxels_list[in * 3 + 1] << "," << voxels_list[in * 3 + 2] << ",";
          for (int scan = K_blk + 1; scan <= row; scan++)
          {
            double *cT_beta_hat = new double[second_col];
            double *beta_hat = new double[col];
            for (int m = 0; m < col; m++)
            {
              beta_hat[m] = _beta_hat[scan - 1][m][pos];
            }
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
                second_col);
            lower[scan - 1] = cT_beta_hat[0] - 1.96 * sqrt(var_cT_beta_hat[scan - 1][i][pos]);
            myfile1 << lower[scan - 1] << ",";
            delete[] cT_beta_hat;
            delete[] beta_hat;
          }
          myfile1 << endl;
          myfile1 << "Upper," << voxels_list[in * 3] << "," << voxels_list[in * 3 + 1] << "," << voxels_list[in * 3 + 2] << ",";
          for (int scan = K_blk + 1; scan <= row; scan++)
          {
            double *cT_beta_hat = new double[second_col];
            double *beta_hat = new double[col];
            for (int m = 0; m < col; m++)
            {
              beta_hat[m] = _beta_hat[scan - 1][m][pos];
            }
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
                second_col);
            upper[scan - 1] = cT_beta_hat[0] + 1.96 * sqrt(var_cT_beta_hat[scan - 1][i][pos]);
            myfile1 << upper[scan - 1] << ",";
            delete[] cT_beta_hat;
            delete[] beta_hat;
          }
          myfile1 << endl;

          myfile1 << "beta_hat," << voxels_list[in * 3] << "," << voxels_list[in * 3 + 1] << "," << voxels_list[in * 3 + 2] << ",";
          for (int scan = K_blk + 1; scan <= row; scan++)
          {
            double *cT_beta_hat = new double[second_col];
            double *beta_hat = new double[col];
            for (int m = 0; m < col; m++)
            {
              beta_hat[m] = _beta_hat[scan - 1][m][pos];
            }
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
                second_col);
            myfile1 << cT_beta_hat[0] << ",";
            delete[] cT_beta_hat;
            delete[] beta_hat;
          }
          myfile1 << endl;

          myfile1 << "Result," << voxels_list[in * 3] << "," << voxels_list[in * 3 + 1] << "," << voxels_list[in * 3 + 2] << ",";
          for (int scan = K_blk + 1; scan <= row; scan++)
          {
            if (_beta_hat[scan - 1][i][pos])
              myfile1 << "-1,";
            else if (_beta_hat[scan - 1][i][pos])
              myfile1 << "1,";
            else
              myfile1 << "0,";
          }
          myfile1 << endl;
        }

        double gamma1 = (1 - beta) / alpha, gamma0 = beta / (1 - alpha);
        double c;
        double P_FA;
        double P_D;
        double E0_K, E1_K, D_p0, D_p1;

        double c_upper = max(gamma0 / 2, gamma1 / 2);
        double c_lower = min(gamma0 / 2, gamma1 / 2);
        myfile1 << "Stopping time estimation (Only applied to unclassified voxels and results can be wild)" << endl;
        myfile1 << "Statistics from scan 119 is used for this estimation" << endl
                << endl;
        myfile1 << "C value lower bound: " << c_lower << ", C value upper bound: " << c_upper << endl;
        for (int in = 0; in < voxels_list.size() / 3; in++)
        {
          myfile1 << "X,Y,Z, ,C,H_0,H_1," << endl;
          int pos = voxels_list[in * 3] * dimensions[1] * dimensions[2] + voxels_list[in * 3 + 1] * dimensions[2] + voxels_list[in * 3 + 2];
          D_p0 = pow(alternative_beta[159][i][pos], 2) / (2 * var_cT_beta_hat[159][i][pos]);
          D_p1 = D_p0;

          // Output specifically when c = 1, where non-log SPRT is neutral
          P_FA = (1 - gamma0) / (gamma1 - gamma0);
          P_D = (gamma1)*P_FA;
          E0_K = (P_FA * log(gamma1) + (1 - P_FA) * log(gamma0)) / (-D_p0);
          E1_K = (P_D * log(gamma1) + (1 - P_D) * log(gamma0)) / (D_p1);
          myfile1 << voxels_list[in * 3] << "," << voxels_list[in * 3 + 1] << "," << voxels_list[in * 3 + 2] << ",," << 1 << "," << E0_K << "," << E1_K << endl;

          c = c_lower;
          while (c < c_upper)
          {
            P_FA = (1 - gamma0 / c) / (gamma1 / c - gamma0 / c);
            P_D = (gamma1 / c) * P_FA;
            E0_K = (P_FA * log(gamma1 / c) + (1 - P_FA) * log(gamma0 / c)) / (-1 * D_p0);
            E1_K = (P_D * log(gamma1 / c) + (1 - P_D) * log(gamma0 / c)) / (D_p1);
            myfile1 << voxels_list[in * 3] << "," << voxels_list[in * 3 + 1] << "," << voxels_list[in * 3 + 2] << ",," << c << "," << E0_K << "," << E1_K << endl;
            if (c < 1)
              c += .1;
            else if (c < 10)
              c += 1;
            else if (c < 100)
              c += 10;
            else if (c < 1000)
              c += 100;
            else
              c += 1000;
          }
        }
        myfile1.close();
      }
    }

    else
      cout << "Error: Coordinates Info Incorrect. Check Your File!";
  }

  /*
   * Output voxel wise activation status
   */
  for (int i = 0; i < num_of_C; i++)
  {
    myfile1.open("Voxel Activation Status for contrast " + to_string(i + 1) + ".csv");
    for (int x = 0; x < dimensions[0]; x++)
    {
      for (int y = 0; y < dimensions[1]; y++)
      {
        for (int z = 0; z < dimensions[2]; z++)
        {
          if (scan_number_for_result_upper[i][x * dimensions[1] * dimensions[2] + y * dimensions[2] + z] == 0 && scan_number_for_result_lower[i][x * dimensions[1] * dimensions[2] + y * dimensions[2] + z] == 0)
            myfile1 << "0,";
          else if (scan_number_for_result_upper[i][x * dimensions[1] * dimensions[2] + y * dimensions[2] + z] == 0 && scan_number_for_result_lower[i][x * dimensions[1] * dimensions[2] + y * dimensions[2] + z] != 0)
            myfile1 << "1,";
          else if (scan_number_for_result_upper[i][x * dimensions[1] * dimensions[2] + y * dimensions[2] + z] != 0 && scan_number_for_result_lower[i][x * dimensions[1] * dimensions[2] + y * dimensions[2] + z] == 0)
            myfile1 << "2,";
          else
            myfile1 << "3,";
        }
        myfile1 << endl;
      }
      myfile1 << endl;
    }
    myfile1.close();
  }

  for (int i = 0; i < num_of_C; i++)
  {
    myfile1.open("Standard_deviation_for_contrast_" + to_string(i + 1) + ".csv");
    myfile1 << ",";
    for (int scan_number = K_blk; scan_number < row; scan_number++)
      myfile1 << "Scan " << to_string(scan_number + 1) << ",";
    myfile1 << endl
            << "Cross, Upper, Bound, Voxels, :" << endl;
    for (int x = 0; x < dimensions[0]; x++)
    {
      for (int y = 0; y < dimensions[1]; y++)
      {
        for (int z = 0; z < dimensions[2]; z++)
        {
          if (is_within_ROI[x * dimensions[1] * dimensions[2] + y * dimensions[2] + z] && scan_number_for_result_upper[i][x * dimensions[1] * dimensions[2] + y * dimensions[2] + z] != 0)
          {
            myfile1 << "[" << setw(3)
                    << x
                    << setw(3)
                    << y
                    << setw(3)
                    << z
                    << "],";
            for (int scan_number = K_blk; scan_number < row; scan_number++)
            {
              myfile1 << sqrt(var_cT_beta_hat[scan_number][i][x * dimensions[1] * dimensions[2] + y * dimensions[2] + z]) << ",";
            }
            myfile1 << endl;
          }
        }
      }
    }

    myfile1 << endl
            << "Cross, Lower, Bound, Voxels, :" << endl;
    for (int x = 0; x < dimensions[0]; x++)
    {
      for (int y = 0; y < dimensions[1]; y++)
      {
        for (int z = 0; z < dimensions[2]; z++)
        {
          if (is_within_ROI[x * dimensions[1] * dimensions[2] + y * dimensions[2] + z] && scan_number_for_result_lower[i][x * dimensions[1] * dimensions[2] + y * dimensions[2] + z] != 0)
          {
            myfile1 << "[" << setw(3)
                    << x
                    << setw(3)
                    << y
                    << setw(3)
                    << z
                    << "],";
            for (int scan_number = K_blk; scan_number < row; scan_number++)
            {
              myfile1 << sqrt(var_cT_beta_hat[scan_number][i][x * dimensions[1] * dimensions[2] + y * dimensions[2] + z]) << ",";
            }
            myfile1 << endl;
          }
        }
      }
    }

    myfile1 << endl
            << "No, Bound, Crossed, Voxels, :" << endl;
    for (int x = 0; x < dimensions[0]; x++)
    {
      for (int y = 0; y < dimensions[1]; y++)
      {
        for (int z = 0; z < dimensions[2]; z++)
        {
          if (is_within_ROI[x * dimensions[1] * dimensions[2] + y * dimensions[2] + z] &&
              scan_number_for_result_upper[i][x * dimensions[1] * dimensions[2] + y * dimensions[2] + z] == 0 &&
              scan_number_for_result_lower[i][x * dimensions[1] * dimensions[2] + y * dimensions[2] + z] == 0)
          {
            myfile1 << "[" << setw(3)
                    << x
                    << setw(3)
                    << y
                    << setw(3)
                    << z
                    << "],";
            for (int scan_number = K_blk; scan_number < row; scan_number++)
            {
              myfile1 << sqrt(var_cT_beta_hat[scan_number][i][x * dimensions[1] * dimensions[2] + y * dimensions[2] + z]) << ",";
            }
            myfile1 << endl;
          }
        }
      }
    }
    myfile1.close();
  }

  for (int i = 0; i < num_of_C; i++)
  {
    myfile1.open("Z_score_for_contrast_" + to_string(i + 1) + ".csv");
    myfile1 << ",";
    for (int scan_number = K_blk; scan_number < row; scan_number++)
      myfile1 << "Scan " << to_string(scan_number + 1) << ",";
    myfile1 << endl
            << "Cross, Upper, Bound, Voxels, :" << endl;
    for (int x = 0; x < dimensions[0]; x++)
    {
      for (int y = 0; y < dimensions[1]; y++)
      {
        for (int z = 0; z < dimensions[2]; z++)
        {
          if (is_within_ROI[x * dimensions[1] * dimensions[2] + y * dimensions[2] + z] && scan_number_for_result_upper[i][x * dimensions[1] * dimensions[2] + y * dimensions[2] + z] != 0)
          {
            myfile1 << "[" << setw(3)
                    << x
                    << setw(3)
                    << y
                    << setw(3)
                    << z
                    << "],";
            for (int scan_number = K_blk; scan_number < row; scan_number++)
            {
              myfile1 << Z_score[scan_number][i][x * dimensions[1] * dimensions[2] + y * dimensions[2] + z] << ",";
            }
            myfile1 << endl;
          }
        }
      }
    }

    myfile1 << endl
            << "Cross, Lower, Bound, Voxels, :" << endl;
    for (int x = 0; x < dimensions[0]; x++)
    {
      for (int y = 0; y < dimensions[1]; y++)
      {
        for (int z = 0; z < dimensions[2]; z++)
        {
          if (is_within_ROI[x * dimensions[1] * dimensions[2] + y * dimensions[2] + z] && scan_number_for_result_lower[i][x * dimensions[1] * dimensions[2] + y * dimensions[2] + z] != 0)
          {
            myfile1 << "[" << setw(3)
                    << x
                    << setw(3)
                    << y
                    << setw(3)
                    << z
                    << "],";
            for (int scan_number = K_blk; scan_number < row; scan_number++)
            {
              myfile1 << Z_score[scan_number][i][x * dimensions[1] * dimensions[2] + y * dimensions[2] + z] << ",";
            }
            myfile1 << endl;
          }
        }
      }
    }

    myfile1 << endl
            << "No, Bound, Crossed, Voxels, :" << endl;
    for (int x = 0; x < dimensions[0]; x++)
    {
      for (int y = 0; y < dimensions[1]; y++)
      {
        for (int z = 0; z < dimensions[2]; z++)
        {
          if (is_within_ROI[x * dimensions[1] * dimensions[2] + y * dimensions[2] + z] &&
              scan_number_for_result_upper[i][x * dimensions[1] * dimensions[2] + y * dimensions[2] + z] == 0 &&
              scan_number_for_result_lower[i][x * dimensions[1] * dimensions[2] + y * dimensions[2] + z] == 0)
          {
            myfile1 << "[" << setw(3)
                    << x
                    << setw(3)
                    << y
                    << setw(3)
                    << z
                    << "],";
            for (int scan_number = K_blk; scan_number < row; scan_number++)
            {
              myfile1 << Z_score[scan_number][i][x * dimensions[1] * dimensions[2] + y * dimensions[2] + z] << ",";
            }
            myfile1 << endl;
          }
        }
      }
    }
    myfile1.close();
  }

  /* delete dynamic allocated memory */
  for (int i = 0; i < row; i++)
  {
    for (int j = 0; j < col; j++)
    {
      delete[] _beta_hat[i][j];
    }
    delete[] _beta_hat[i];
  }
  delete[] _beta_hat;

  delete[] XTX_inverse;
  delete[] X;

  // delete 3 dimension array
  for (int i = 0; i < row; i++)
  {
    for (int j = 0; j < num_of_C; j++)
    {
      delete[] Z_score[i][j];
      delete[] var_cT_beta_hat[i][j];
      delete[] SPRT[i][j];
      delete[] alternative_beta[i][j];
    }
    delete[] Z_score[i];
    delete[] var_cT_beta_hat[i];
    delete[] SPRT[i];
    delete[] alternative_beta[i];
  }
  delete[] var_cT_beta_hat;
  delete[] Z_score;
  delete[] SPRT;
  delete[] alternative_beta;

  // delete 2d array
  for (int i = 0; i < num_of_C; i++)
  {
    delete[] C[i];
    delete[] scan_number_for_result_lower[i];
    delete[] scan_number_for_result_upper[i];
  }
  delete[] C;
  delete[] scan_number_for_result_lower;
  delete[] scan_number_for_result_upper;

  cout << ".  .  ." << endl
       << ".  .  ." << endl
       << ".  .  ." << endl
       << "SPRT analysis finished." << endl;

  return 0;
}