check_package_installed <- function(pkg) {
    if (!requireNamespace(pkg, quietly = TRUE)) {
        stop(paste("The package", pkg, "is not installed. Please install it and try again."))
    }
}

check_package_installed('dplyr')
library(dplyr)

# Do a lookup to see if there are responses missing for one or more of those trials
# If one missing, assign to that trial 
# If both missing, assign to current trial (unless it's a nogo trial)


get_columns <- function(d) {
    # Need trial, condition, coherence, and rt
    if (!'rt' %in% colnames(d)) {
        stop('Need a column in the data labeled "rt"')
    }
    
    rt <- d$rt
    trial <- colnames(d)[sapply(colnames(d), FUN = function(x) grepl('trial', x))]
    if (identical(character(0), trial)) {
        stop('Need a column with the word "trial" in the label.')
    }
    trial <- d[,trial]
    if (!'condition' %in% colnames(d)) {
        stop('Need a column named "condition" storing the trial type as either "dom" or "nondom"')
    }
    condition <- d$condition
    if (!'coherence' %in% colnames(d)) {
        stop('Need a column named "coherence" storing the coherence of the stimulus as a numeric between 0 and 1.')
    }
    coherence <- d$coherence
    
    if (!'resp_key' %in% colnames(d)) {
        stop('Need a column named "resp_key" storing the response key.')
    }
    
    out <- data.frame(trial=trial, condition=condition, coherence=coherence, rt=rt)
    
    return(out)
    
}



assign_response <- function(d, unambig_low = .4, unambig_high = .55) {
    # --- PARAMETERS --- #
    # d:            GradCPT data from one subject and one run/block
    # unambig_low:  lower limit for unambiguous response assignment
    # unambig_high: upper limit for unambiguous response assignment
    #
    # RETURNS
    # data frame with each response assigned to a trial and coded accuracy
    
    # Check incoming data
    stopifnot(is.data.frame(d))    
    
    # Get relevant data
    temp <- get_columns(d)
    trial <- temp$trial
    condition <- temp$condition
    coherence <- temp$coherence
    rt <- temp$rt
    
    within_trial <- c('total_runtime_mins', 'frame_count', 'coherence', 'resp_key', 'rt')
    
    if (! all(within_trial %in% colnames(d))) {
        stop(paste0('The following columns need to be in the data: ', paste0(within_trial, collapse=', ')))
    }
    
    # Step 1 - Assign unambiguous responses
    trialcode1 <- ifelse(rt == 0, 'timeout', 
                        ifelse(coherence < unambig_low, lag(trial), 
                        ifelse(coherence > unambig_high, trial, 'ambiguous')))
    
    # Step 2 - Assign ambiguous responses
    trialcode2 <- c()
    
    for (row in 1:(length(rt))) {
        
        # If current response assignment is unambiguous, leave it alone
        if (trialcode1[row] != 'ambiguous') {
            trialcode2 <- c(trialcode2, trialcode1[row])
            
        # If current response assignment is ambiguous
        } else if (trialcode1[row] == 'ambiguous') {
           trial_id <- trial[row]
           # Discern if the current and previous trials are already assigned a response
           
           # To discern previous trial:
           # If it's not the first iteration, look back to the 
           # last element in the currently created step 2 vector 
           if (length(trialcode2 > 2)) {
               is_previous_response <- as.character(trial_id - 1) %in% trialcode2[row-1]
           # Else, just look in the step 1 results
           } else {
               is_previous_response <- as.character(trial_id - 1) %in% trialcode1
           }
           # Discern whether current trial is assigned a response
           is_current_response <- as.character(trial_id) %in% trialcode1
           
           # If exactly one trial is assigned a response
           if (xor(is_previous_response, is_current_response)) {
               # Assign the current response to the trial that isn't already assigned
               trialcode2 <- c(trialcode2, ifelse(is_previous_response, trial_id, trial_id-1))
           # If neither current or previous trial is assigned a response
           } else if (!is_previous_response & !is_current_response) {
               # Assign it to the current trial only if current trial is not catch trial
               trialcode2 <- c(trialcode2, ifelse(condition[row]=='nondom', trial_id - 1, trial_id))
           # If both trials are already assigned, assign it as duplicate to 
           # current trial
           } else{
               trialcode2 <- c(trialcode2, trial_id)
           }
        } 
        
    } # End Step 2
    
    # -- Step 3 - keep only first of duplicate responses --
    
    # Format results from Step 2 as data frame
    s2 <- cbind(data.frame(trial=trial, condition=condition), d[,colnames(d) %in% within_trial], trialcode2)
    # Ensure trial column is named trial
    colnames(d)[sapply(colnames(d), FUN=function(x) grepl('trial', x))] <- 'trial'
    
    # Keep only non duplicate observations for within subject columns
    coded <- s2 %>% 
        # Recode RTs assigned to previous trial to add 800 ms
        mutate(new_rt = ifelse(trialcode2 == lag(trial) & trialcode2 != trial, .8+rt, rt),
               # Identify duplicated trial assignments
               dups = duplicated(trialcode2)) %>% 
        filter(trialcode2 != 'timeout', !dups) %>% 
        # Format columns
        select(trialcode2, all_of(within_trial[within_trial != 'rt']), new_rt) %>% 
        rename(trial = trialcode2, rt = new_rt) %>% 
        mutate(trial = as.numeric(trial), rt = as.numeric(rt)) 
    
    # Extract all between subject information and merge it to within trial data
    d <- d %>% 
        mutate(dups = duplicated(trial)) %>% 
        filter(!dups) %>% 
        select(-all_of(within_trial), -dups) %>% 
        # Left join ensures missing trials are coded as timeouts
        left_join(coded) 
    
    
    # Finally, code accuracy
    
    d <- d %>% 
        mutate(accuracy = case_when(
            condition == 'nondom' & is.na(rt) ~ 1,
            condition == 'dom' & !is.na(rt) ~ 1,
            TRUE ~ 0
        )) 
    
    return(d)
}


