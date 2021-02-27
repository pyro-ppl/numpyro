functions {

    // function equivalent to %in% on R from https://discourse.mc-stan.org/t/stan-equivalent-of-rs-in-function/3849
    int r_in(int pos,int[] pos_var) {
   
        for (p in 1:(size(pos_var))) {
            if (pos_var[p]==pos) {
                // can return immediately, as soon as find a match
                return 1;
            }
        }
        return 0;
    }

    /*
     * returns multiplier on the rows of the contact matrix over time for one country
     */
    matrix country_impact(
        vector beta,
        real dip_rdeff_local,
        real[] upswing_timeeff_reduced_local,
        int N2,
        int A,
        int A_CHILD,
        int[] AGE_CHILD,
        int COVARIATES_N,
        matrix[] covariates_local,
        row_vector upswing_age_rdeff_local,
        int[] upswing_timeeff_map_local
        )
    {
        // scaling of contacts after intervention effect on day t in location m
        matrix[N2,A] impact_intv;
    
        // define multipliers for contacts in each location
        impact_intv = (beta[2] + rep_matrix(upswing_age_rdeff_local, N2)) .* covariates_local[3];
        
        // expand upswing time effect
        impact_intv .*= rep_matrix( to_vector( upswing_timeeff_reduced_local[ upswing_timeeff_map_local ]), A );
        
        // add other coeff*predictors
        impact_intv += covariates_local[1];
        impact_intv += ( beta[1] + dip_rdeff_local ) * covariates_local[2];
        
        impact_intv = exp( impact_intv );
        
        // impact_intv set to 1 for children
        impact_intv[:,AGE_CHILD] = rep_matrix(1.0, N2, A_CHILD);
        
        return (impact_intv);
    }
  
    matrix country_EcasesByAge(// parameters
        real R0_local,
        real e_cases_N0_local,
        row_vector log_relsusceptibility_age,
        real impact_intv_children_effect,
        real impact_intv_onlychildren_effect,
        matrix impact_intv,
        // data
        int N0,
        int elementary_school_reopening_idx_local,
        int N2,
        vector SCHOOL_STATUS_local,
        int A,
        int A_CHILD,
        int SI_CUT,
        int[] wkend_idx_local,
        real avg_cntct_local,
        matrix cntct_weekends_mean_local,
        matrix cntct_weekdays_mean_local,
        matrix cntct_school_closure_weekends_local,
        matrix cntct_school_closure_weekdays_local,
        matrix cntct_elementary_school_reopening_weekends_local,
        matrix cntct_elementary_school_reopening_weekdays_local,
        row_vector rev_serial_interval,
        row_vector popByAge_abs_local,
        int N_init_A,
        int[] init_A
        )
    {
        real zero = 0.0;
        real N_init_A_real= N_init_A*1.;
        row_vector[A] tmp_row_vector_A;
        row_vector[A] tmp_row_vector_A_no_impact_intv;
        row_vector[A] tmp_row_vector_A_with_children_impact_intv;
        row_vector[A] tmp_row_vector_A_with_children_and_childrenchildren_impact_intv;
        
        // probability of infection given contact in location m
        real rho0 = R0_local / avg_cntct_local;
          
        // expected new cases by calendar day, age, and location under self-renewal model
        // and a container to store the precomputed cases by age
        matrix[N2,A] E_casesByAge = rep_matrix( zero, N2, A );
          
        // init expected cases by age and location in first N0 days
        E_casesByAge[1:N0, init_A] = rep_matrix( e_cases_N0_local/N_init_A_real, N0, N_init_A);
        
        // calculate expected cases by age and country under self-renewal model after first N0 days
        // and adjusted for saturation
        E_casesByAge += 1000;
        /*
        for (t in (N0+1):N2)
        {
            int start_idx_rev_serial = SI_CUT-t+2;
            int start_idx_E_casesByAge = t-SI_CUT;
            row_vector[A] prop_susceptibleByAge = rep_row_vector(1.0, A) - (rep_row_vector(1.0, t-1) * E_casesByAge[1:(t-1),:] ./ popByAge_abs_local);
            
            if(start_idx_rev_serial < 1)
            {
                start_idx_rev_serial = 1;
            }
            if(start_idx_E_casesByAge < 1)
            {
                start_idx_E_casesByAge = 1;
            }
            // TODO can t we vectorise this?
            for(a in 1:A)
            {
                if(prop_susceptibleByAge[a] < 0)
                { // account for values of Ecases > pop at initalization
                    prop_susceptibleByAge[a] = 0;
                }
            }
            
            tmp_row_vector_A = rev_serial_interval[start_idx_rev_serial:SI_CUT] * E_casesByAge[start_idx_E_casesByAge:(t-1)];
            tmp_row_vector_A *= rho0;
            tmp_row_vector_A_no_impact_intv = tmp_row_vector_A;
            tmp_row_vector_A .*= impact_intv[t,];
            
            if(r_in(t, wkend_idx_local) == 1)
            {
                // school open
                if(SCHOOL_STATUS_local[t] == 0 && t<elementary_school_reopening_idx_local)
                {
                    E_casesByAge[t] =
                        append_col(
                            tmp_row_vector_A_no_impact_intv * cntct_weekends_mean_local[,1:A_CHILD],
                            tmp_row_vector_A_no_impact_intv[1:A_CHILD] * cntct_weekends_mean_local[1:A_CHILD,(A_CHILD+1):A] +
                                (tmp_row_vector_A[(A_CHILD+1):A] * cntct_weekends_mean_local[(A_CHILD+1):A,(A_CHILD+1):A]) .* impact_intv[t,(A_CHILD+1):A]
                            );
                }
                // school reopen
                else if(SCHOOL_STATUS_local[t] == 0 && t>=elementary_school_reopening_idx_local)
                {
                    tmp_row_vector_A_with_children_impact_intv = tmp_row_vector_A;
                    tmp_row_vector_A_with_children_impact_intv[1:A_CHILD] *= impact_intv_children_effect;
                    tmp_row_vector_A_with_children_and_childrenchildren_impact_intv = tmp_row_vector_A_with_children_impact_intv;
                    tmp_row_vector_A_with_children_and_childrenchildren_impact_intv[1:A_CHILD] *= impact_intv_onlychildren_effect;
                    
                    E_casesByAge[t] =
                        append_col(
                            tmp_row_vector_A_with_children_and_childrenchildren_impact_intv * cntct_elementary_school_reopening_weekends_local[,1:A_CHILD],
                            tmp_row_vector_A_with_children_impact_intv * cntct_elementary_school_reopening_weekends_local[,(A_CHILD+1):A]
                            );
                    E_casesByAge[t, 1:A_CHILD] *= impact_intv_children_effect;
                    E_casesByAge[t, (A_CHILD+1):A] .*= impact_intv[t,(A_CHILD+1):A];
                }
                // school closed
                else
                {
                    E_casesByAge[t] =
                        append_col(
                            tmp_row_vector_A_no_impact_intv * cntct_school_closure_weekends_local[,1:A_CHILD],
                            tmp_row_vector_A_no_impact_intv[1:A_CHILD] * cntct_school_closure_weekends_local[1:A_CHILD,(A_CHILD+1):A] +
                                (tmp_row_vector_A[(A_CHILD+1):A] * cntct_school_closure_weekends_local[(A_CHILD+1):A,(A_CHILD+1):A]) .* impact_intv[t,(A_CHILD+1):A]
                            );
                }
            }
            else
            {
                if(SCHOOL_STATUS_local[t] == 0 && t<elementary_school_reopening_idx_local)
                {
                    E_casesByAge[t] =
                        append_col(
                            tmp_row_vector_A_no_impact_intv * cntct_weekdays_mean_local[,1:A_CHILD],
                            tmp_row_vector_A_no_impact_intv[1:A_CHILD] * cntct_weekdays_mean_local[1:A_CHILD,(A_CHILD+1):A] +
                                (tmp_row_vector_A[(A_CHILD+1):A] * cntct_weekdays_mean_local[(A_CHILD+1):A,(A_CHILD+1):A]) .* impact_intv[t,(A_CHILD+1):A]
                            );
                }
                else if(SCHOOL_STATUS_local[t] == 0 && t>=elementary_school_reopening_idx_local)
                {
                    tmp_row_vector_A_with_children_impact_intv = tmp_row_vector_A;
                    tmp_row_vector_A_with_children_impact_intv[1:A_CHILD] *= impact_intv_children_effect;
                    tmp_row_vector_A_with_children_and_childrenchildren_impact_intv = tmp_row_vector_A_with_children_impact_intv;
                    tmp_row_vector_A_with_children_and_childrenchildren_impact_intv[1:A_CHILD] *= impact_intv_onlychildren_effect;
                    
                    E_casesByAge[t] =
                        append_col(
                            tmp_row_vector_A_with_children_and_childrenchildren_impact_intv * cntct_elementary_school_reopening_weekdays_local[,1:A_CHILD],
                            tmp_row_vector_A_with_children_impact_intv * cntct_elementary_school_reopening_weekdays_local[,(A_CHILD+1):A]
                            );
                    E_casesByAge[t, 1:A_CHILD] *= impact_intv_children_effect;
                    E_casesByAge[t, (A_CHILD+1):A] .*= impact_intv[t,(A_CHILD+1):A];
                }
                else
                {
                    E_casesByAge[t] =
                        append_col(
                            tmp_row_vector_A_no_impact_intv * cntct_school_closure_weekdays_local[,1:A_CHILD],
                            tmp_row_vector_A_no_impact_intv[1:A_CHILD] * cntct_school_closure_weekdays_local[1:A_CHILD,(A_CHILD+1):A] +
                                (tmp_row_vector_A[(A_CHILD+1):A] * cntct_school_closure_weekdays_local[(A_CHILD+1):A,(A_CHILD+1):A]) .* impact_intv[t,(A_CHILD+1):A]
                            );
                }
            }
            
            E_casesByAge[t] .*= prop_susceptibleByAge;
            E_casesByAge[t] .*= exp(log_relsusceptibility_age);

        }
        */
        return(E_casesByAge);
    }
  
    matrix country_EdeathsByAge(// parameters
        matrix E_casesByAge_local,
        // data
        int N2,
        int A,
        row_vector rev_ifr_daysSinceInfection,
        row_vector log_ifr_age_base,
        real log_ifr_age_rnde_mid1_local,
        real log_ifr_age_rnde_mid2_local,
        real log_ifr_age_rnde_old_local
        )
    {
        real zero = 0.0;
        
        matrix[N2,A] E_deathsByAge = rep_matrix( zero, N2, A );
    
        // calculate expected deaths by age and country
        E_deathsByAge[1] = 1e-15 * E_casesByAge_local[1];
        for (t in 2:N2)
        {
            E_deathsByAge[t] = rev_ifr_daysSinceInfection[(N2-(t-1)+1):N2 ] * E_casesByAge_local[1:(t-1)];
        }
        E_deathsByAge .*= rep_matrix(exp(   log_ifr_age_base +
            append_col(append_col(append_col(
                rep_row_vector(0., 4),
                rep_row_vector(log_ifr_age_rnde_mid1_local, 6)),
                rep_row_vector(log_ifr_age_rnde_mid2_local, 4)),
                rep_row_vector(log_ifr_age_rnde_old_local, 4))
            ), N2);
  
        E_deathsByAge += 1e-15;
        return(E_deathsByAge);
    }

    real countries_log_dens(int[,] deaths_slice,
        int start,
        int end,
        // parameters
        vector R0,
        real[] e_cases_N0,
        vector beta,
        real[] dip_rdeff,
        real[,] upswing_timeeff_reduced,
        matrix timeeff_shift_age,
        row_vector log_relsusceptibility_age,
        real phi,
        real impact_intv_children_effect,
        real impact_intv_onlychildren_effect,
        // data
        int N0,
        int[] elementary_school_reopening_idx,
        int N2,
        matrix SCHOOL_STATUS,
        int A,
        int A_CHILD,
        int[] AGE_CHILD,
        int COVARIATES_N,
        int SI_CUT,
        int[] num_wkend_idx,
        int[,] wkend_idx,
        int[,] upswing_timeeff_map,
        vector avg_cntct,
        matrix[,] covariates,
        matrix[] cntct_weekends_mean,
        matrix[] cntct_weekdays_mean,
        matrix[] cntct_school_closure_weekends,
        matrix[] cntct_school_closure_weekdays,
        matrix[] cntct_elementary_school_reopening_weekends,
        matrix[] cntct_elementary_school_reopening_weekdays,
        row_vector rev_ifr_daysSinceInfection,
        row_vector log_ifr_age_base,
        row_vector log_ifr_age_rnde_mid1,
        row_vector log_ifr_age_rnde_mid2,
        row_vector log_ifr_age_rnde_old,
        row_vector rev_serial_interval,
        int[] epidemicStart,
        int[] N,
        int N_init_A,
        int[] init_A,
        int[] A_AD,
        int[] dataByAgestart,
        matrix[] map_age,
        int[,,] deathsByAge,
        int[,] map_country,
        matrix popByAge_abs,
        vector ones_vector_A,
        int[] smoothed_logcases_weeks_n,
        int[,,] smoothed_logcases_week_map,
        real[,,] smoothed_logcases_week_pars,
        int[,] school_case_time_idx,
        real[,] school_case_data
        )
  {
    real lpmf = 0.0;
    int M_slice = end - start + 1;
    int index_country_slice; 
    int min_age_slice;
    int max_age_slice;
    vector[N2] E_cases;
    vector[N2] E_deaths;
    matrix[N2, A] impact_intv;
    matrix[N2, A] E_casesByAge;
    matrix[N2, A] E_deathsByAge;
    real school_attack_rate;
    // matrix[N2, A] Rt_byAge;

    for(m_slice in 1:M_slice) {
      int m = m_slice + start - 1;
      real E_log_week_avg_cases[smoothed_logcases_weeks_n[m]];
        
      impact_intv =
        country_impact(
            beta,
            dip_rdeff[m],
            upswing_timeeff_reduced[,m],
            N2,
            A,
            A_CHILD,
            AGE_CHILD,
            COVARIATES_N,
            covariates[m],
            timeeff_shift_age[m,],
            upswing_timeeff_map[,m]
            );
      
      E_casesByAge =
        country_EcasesByAge(
            R0[m],
            e_cases_N0[m],
            log_relsusceptibility_age,
            impact_intv_children_effect,
            impact_intv_onlychildren_effect,
            impact_intv,
            N0,
            elementary_school_reopening_idx[m],
            N2,
            SCHOOL_STATUS[,m],
            A,
            A_CHILD,
            SI_CUT,
            wkend_idx[1:num_wkend_idx[m],m],
            avg_cntct[m],
            cntct_weekends_mean[m],
            cntct_weekdays_mean[m],
            cntct_school_closure_weekends[m],
            cntct_school_closure_weekdays[m],
            cntct_elementary_school_reopening_weekends[m],
            cntct_elementary_school_reopening_weekdays[m],
            rev_serial_interval,
            popByAge_abs[m],
            N_init_A,
            init_A
            );
      
      
      E_deathsByAge =
        country_EdeathsByAge(
            E_casesByAge,
            N2,
            A,
            rev_ifr_daysSinceInfection,
            log_ifr_age_base,
            log_ifr_age_rnde_mid1[m],
            log_ifr_age_rnde_mid2[m],
            log_ifr_age_rnde_old[m]
            );
            
      E_cases = E_casesByAge * ones_vector_A;
      
      E_deaths = E_deathsByAge * ones_vector_A;
      
      //
      // likelihood death data this location
      //
      if(map_country[m,1] == 1){
        index_country_slice = map_country[m,2];
        lpmf += neg_binomial_2_lpmf(deaths_slice[m_slice, epidemicStart[m]:(dataByAgestart[index_country_slice]-1)] | E_deaths[epidemicStart[m]:(dataByAgestart[index_country_slice]-1)], phi );
        
        for(a in 1:A_AD[index_country_slice]){
          // first day of data is sumulated death
          lpmf += neg_binomial_2_lpmf(deathsByAge[dataByAgestart[index_country_slice], a, index_country_slice] | 
                                          rep_row_vector(1.0, (dataByAgestart[index_country_slice]-epidemicStart[m]+1)) * E_deathsByAge[epidemicStart[m]:dataByAgestart[index_country_slice], :] * map_age[index_country_slice][:, a], phi );
          // after daily death
          lpmf += neg_binomial_2_lpmf(deathsByAge[(dataByAgestart[index_country_slice]+1):N[m], a, index_country_slice] | 
                                          E_deathsByAge[(dataByAgestart[index_country_slice]+1):N[m], :] * map_age[index_country_slice][:, a], phi );
          }
      }
      if(map_country[m,1] == 0){
        lpmf += neg_binomial_2_lpmf(deaths_slice[m_slice, epidemicStart[m]:N[m]]| E_deaths[epidemicStart[m]:N[m]], phi );
      }
      
      //
      // likelihood case data this location
      //
      for( w in 1:smoothed_logcases_weeks_n[m])
      {
        E_log_week_avg_cases[w] = mean( log( E_cases[ smoothed_logcases_week_map[m, w, :] ] ) );
      }
      lpmf += student_t_lcdf( E_log_week_avg_cases |
        smoothed_logcases_week_pars[m, 1:smoothed_logcases_weeks_n[m], 3],
        smoothed_logcases_week_pars[m, 1:smoothed_logcases_weeks_n[m], 1],
        smoothed_logcases_week_pars[m, 1:smoothed_logcases_weeks_n[m], 2]
        );
      
      //
      // likelihood school case data this location
      //
      if( school_case_time_idx[m,1]>0 )
      {
        school_attack_rate = sum(append_col(
            E_casesByAge[ school_case_time_idx[m,1]:school_case_time_idx[m,2], 2:3 ],
            0.8*E_casesByAge[ school_case_time_idx[m,1]:school_case_time_idx[m,2], 4 ]
            ));
        school_attack_rate /= sum( [ popByAge_abs[m, 2], popByAge_abs[m, 3], 0.8 * popByAge_abs[m, 4 ] ] );
        
        // prevent over/underflow
        school_attack_rate = min([ school_attack_rate, school_case_data[m,3]*4]);
        
        lpmf += normal_lcdf( school_attack_rate | school_case_data[m,1], school_case_data[m,2]);
        lpmf += normal_lcdf( -school_attack_rate | -school_case_data[m,3], school_case_data[m,4]);
      }
    }
    return(lpmf);
  }
}

data {
  int<lower=1> M; // number of countries
  int<lower=1> N0; // number of initial days for which to estimate infections
  int<lower=1> N[M]; // days of observed data for country m. each entry must be <= N2
  int<lower=1> N2; // days of observed data + # of days to forecast
  int<lower=1> A; // number of age bands
  int<lower=1> SI_CUT; // number of days in serial interval to consider
  int<lower=1> COVARIATES_N; // number of days in serial interval to consider
  int<lower=1>  N_init_A; // number of age bands with initial cases
  int WKEND_IDX_N[M]; // number of weekend indices in each location
  int<lower=1, upper=N2> N_IMP; // number of impact invervention time effects
  //	data
  real pop[M];
  matrix<lower=0, upper=1>[A,M] popByAge; // proportion of age bracket in population in location
  int epidemicStart[M];
  int deaths[N2, M]; // reported deaths -- the rows with i > N contain -1 and should be ignored
  int<lower=1> elementary_school_reopening_idx[M]; // time index after which schools reopen for country m IF school status is set to 0. each entry must be <= N2
  int<lower=0> wkend_idx[N2,M]; //indices of 1:N2 that correspond to weekends in location m
  int<lower=1,upper=N_IMP> upswing_timeeff_map[N2,M]; // map of impact intv time effects to time units in model for each state
  //
  // mobility trends
  //
  matrix[N2,A] covariates[M, COVARIATES_N]; // predictors for fsq contacts by age
  //
  // death data by age
  //
  int<lower=0> M_AD; // number of countries with deaths by age data
  int<lower=1> dataByAgestart[M_AD]; // start of death by age data
  int deathsByAge[N2, A, M_AD]; // reported deaths by age -- the rows with i < dataByAgestart[M_AD] contain -1 and should be ignored + the column with j > A2[M_AD] contain -1 and should be ignored 
  int<lower=2> A_AD[M_AD]; // number of age groups reported 
  matrix[A, A] map_age[M_AD]; // map the age groups reported with 5 y age group -- the column with j > A2[M_AD] contain -1 and should be ignored
  int map_country[M,2]; // first column indicates if country has death by age date (1 if yes), 2 column map the country to M_AD
  //
  // case data by age
  //
  int smoothed_logcases_weeks_n_max;
  int<lower=1, upper=smoothed_logcases_weeks_n_max> smoothed_logcases_weeks_n[M]; // number of week indices per location
  int smoothed_logcases_week_map[M, smoothed_logcases_weeks_n_max, 7]; // map of week indices to time indices
  real smoothed_logcases_week_pars[M, smoothed_logcases_weeks_n_max, 3]; // likelihood parameters for observed cases
  //
  // school case data
  //
  int school_case_time_idx[M,2];
  real school_case_data[M,4];
  //
  // school closure status
  //
  matrix[N2,M] SCHOOL_STATUS; // school status, 1 if close, 0 if open
  //
  // contact matrices
  //
  int A_CHILD; // number of age band for child
  int<lower=1,upper=A> AGE_CHILD[A_CHILD]; // age bands with child
  matrix[A,A] cntct_school_closure_weekdays[M]; // min cntct_weekdays_mean and contact intensities during outbreak estimated in Zhang et al 
  matrix[A,A] cntct_school_closure_weekends[M]; // min cntct_weekends_mean and contact intensities during outbreak estimated in Zhang et al
  matrix[A,A] cntct_elementary_school_reopening_weekdays[M]; // contact matrix for school reopening
  matrix[A,A] cntct_elementary_school_reopening_weekends[M]; // contact matrix for school reopening
  //	priors
  matrix[A,A] cntct_weekdays_mean[M]; // mean of prior contact rates between age groups on weekdays
  matrix[A,A] cntct_weekends_mean[M]; // mean of prior contact rates between age groups on weekends
  real<upper=0> hyperpara_ifr_age_lnmu[A];  // hyper-parameters for probability of death in age band a log normal mean
  real<lower=0> hyperpara_ifr_age_lnsd[A];  // hyper-parameters for probability of death in age band a log normal sd
  row_vector[N2] rev_ifr_daysSinceInfection; // probability of death s days after infection in reverse order
  row_vector[SI_CUT] rev_serial_interval; // fixed pre-calculated serial interval using empirical data from Neil in reverse order
  int<lower=1, upper=A> init_A[N_init_A]; // age band in which initial cases occur in the first N0 days
}

transformed data {
  vector<lower=0>[M] avg_cntct;
  vector[A] ones_vector_A = rep_vector(1.0, A);
  row_vector[A] ones_row_vector_A = rep_row_vector(1.0, A);
  int trans_deaths[M, N2]; // reported deaths -- the rows with i > N contain -1 and should be ignored
  matrix[M,A] popByAge_abs;  
  
  for( m in 1:M )
  {
    avg_cntct[m] = popByAge[:,m]' * ( cntct_weekdays_mean[m] * ones_vector_A ) * 5./7.;
    avg_cntct[m] += popByAge[:,m]' * ( cntct_weekends_mean[m] * ones_vector_A ) * 2./7.;

    trans_deaths[m,:] = deaths[:,m];
    
    popByAge_abs[m,:] = popByAge[:,m]' * pop[m]; // pop by age is a proportion of pop by age and pop is the absolute number 
  }
}

parameters {
  vector<lower=0>[M] R0; // R0
  real<lower=0> e_cases_N0[M]; // expected number of cases per day in the first N0 days, for each country
  vector[COVARIATES_N-1] beta; // regression coefficients for time varying multipliers on contacts
  real dip_rnde[M];
  real<lower=0> sd_dip_rnde;
  real<lower=0> phi; // overdispersion parameter for likelihood model
  row_vector<upper=0>[A] log_ifr_age_base; // probability of death for age band a
  real<lower=0> hyper_log_ifr_age_rnde_mid1;
  real<lower=0> hyper_log_ifr_age_rnde_mid2;
  real<lower=0> hyper_log_ifr_age_rnde_old;
  row_vector<lower=0>[M] log_ifr_age_rnde_mid1;
  row_vector<lower=0>[M] log_ifr_age_rnde_mid2;
  row_vector<lower=0>[M] log_ifr_age_rnde_old;
  row_vector[2] log_relsusceptibility_age_reduced;
  real<lower=0> upswing_timeeff_reduced[N_IMP,M];
  real<lower=0> sd_upswing_timeeff_reduced;
  real<lower=0> hyper_timeeff_shift_mid1;
  vector<lower=0>[M] timeeff_shift_mid1;
  real<lower=0.1,upper=1.0> impact_intv_children_effect;
  real<lower=0> impact_intv_onlychildren_effect;
}

transformed parameters {
  row_vector[A] log_relsusceptibility_age =
    append_col(
        append_col(
            log_relsusceptibility_age_reduced[ { 1, 1, 1 } ],
            rep_row_vector(0., 10)
            ),
        log_relsusceptibility_age_reduced[ { 2,2,2,2,2 } ]
        );
    
  matrix[M,A] timeeff_shift_age =
    append_col(
        append_col(
            rep_matrix(0., M, 4),
            rep_matrix(timeeff_shift_mid1, 6)
            ),
        rep_matrix(0., M, 8)
    );
}

model {
  // priors
  target += lognormal_lpdf( e_cases_N0 | 4.85, 0.4); // qlnorm(c(0.01,0.5,0.99), 4.85, 0.4) = 50.37328 127.74039 323.93379
  target += normal_lpdf( phi | 0, 5);
  target += normal_lpdf( beta | 0, 1);
  target += lognormal_lpdf(R0 | 0.98, 0.2); // qlnorm(c(0.025,0.5,0.975), .98, 0.2) = 1.800397 2.664456 3.943201
  target += exponential_lpdf( hyper_log_ifr_age_rnde_mid1 | .1);
  target += exponential_lpdf( hyper_log_ifr_age_rnde_mid2 | .1);
  target += exponential_lpdf( hyper_log_ifr_age_rnde_old | .1);
  target += exponential_lpdf( log_ifr_age_rnde_mid1 | hyper_log_ifr_age_rnde_mid1);
  target += exponential_lpdf( log_ifr_age_rnde_mid2 | hyper_log_ifr_age_rnde_mid2);
  target += exponential_lpdf(log_ifr_age_rnde_old | hyper_log_ifr_age_rnde_old);
  target += normal_lpdf(log_ifr_age_base | hyperpara_ifr_age_lnmu, hyperpara_ifr_age_lnsd);
  target += normal_lpdf( log_relsusceptibility_age_reduced[1] | -1.0702331, 0.2169696);//citation: Zhang et al Science
  target += normal_lpdf( log_relsusceptibility_age_reduced[2] | 0.3828269, 0.1638433);//citation: Zhang et al Science
  target += exponential_lpdf( sd_dip_rnde | 1.5);
  target += normal_lpdf( dip_rnde | 0, sd_dip_rnde);
  target += lognormal_lpdf( sd_upswing_timeeff_reduced | -1.2, 0.2); // qlnorm(c(0.01,0.5,0.99), -1.2, .2) = 0.1891397 0.3011942 0.4796347
  target += normal_lpdf(upswing_timeeff_reduced[1,:] | 0, 0.025);
  target += normal_lpdf( to_array_1d(upswing_timeeff_reduced[2:N_IMP,:]) | to_array_1d(upswing_timeeff_reduced[1:(N_IMP-1),:]), sd_upswing_timeeff_reduced);
  target += exponential_lpdf( hyper_timeeff_shift_mid1 | .1);
  target += exponential_lpdf( timeeff_shift_mid1 | hyper_timeeff_shift_mid1);
  target += lognormal_lpdf( impact_intv_onlychildren_effect | 0, 0.35); // exp(-0.35*2) = 0.496
  
  // rstan version
  // target += countries_log_dens(trans_deaths, 1, M,
  // cmdstan version
  target += reduce_sum(countries_log_dens, trans_deaths, 1,
      R0,
      e_cases_N0,
      beta,
      dip_rnde,
      upswing_timeeff_reduced,
      timeeff_shift_age,
      log_relsusceptibility_age,
      phi,
      impact_intv_children_effect,
      impact_intv_onlychildren_effect,
      N0,
      elementary_school_reopening_idx,
      N2,
      SCHOOL_STATUS,
      A,
      A_CHILD,
      AGE_CHILD,
      COVARIATES_N,
      SI_CUT,
      WKEND_IDX_N,
      wkend_idx,
      upswing_timeeff_map,
      avg_cntct,
      covariates,
      cntct_weekends_mean,
      cntct_weekdays_mean,
      cntct_school_closure_weekends,
      cntct_school_closure_weekdays,
      cntct_elementary_school_reopening_weekends,
      cntct_elementary_school_reopening_weekdays,
      rev_ifr_daysSinceInfection,
      log_ifr_age_base,
      log_ifr_age_rnde_mid1,
      log_ifr_age_rnde_mid2,
      log_ifr_age_rnde_old,
      rev_serial_interval,
      epidemicStart,
      N,
      N_init_A,
      init_A,
      A_AD,
      dataByAgestart,
      map_age,
      deathsByAge,
      map_country,
      popByAge_abs,
      ones_vector_A,
      smoothed_logcases_weeks_n,
      smoothed_logcases_week_map,
      smoothed_logcases_week_pars,
      school_case_time_idx,
      school_case_data
      );
}
