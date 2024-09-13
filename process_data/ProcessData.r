path <- commandArgs(trailingOnly = TRUE)

if (!grepl('gradcptpy', getwd())) {
    stop('Need to run this script from inside the GradCPTpy repo.')
}

infer_path <- function() {
    # Change wd to root directory

    wd <- tolower(getwd())
    
    if (basename(wd) != 'gradcptpy') {
        path_components <- strsplit(wd, '/')[[1]]
        index <- which(path_components == 'gradcptpy')
        steps_back <- length(path_components) - index
        new_wd <- rep('../', steps_back)
        setwd(new_wd)
    }

}

og_wd <- getwd()
infer_path()
source('process_data/concatentate_gradcpt.r')
source('process_data/response_assignment.r')

if (length(path) != 1) {
    message('No path supplied. Changing working directory to root of repo.')
    path <- 'neat_data'
} else {
    setwd(og_wd)
}


is_directory <- function(path) {
    info <- file.info(path)
    isdir <- ifelse(!is.na(info$isdir) & info$isdir, TRUE, FALSE)
    return(isdir)
}

is_csv_file <- function(path) {
    return(file.exists(path) && grepl('\\.csv', path, ignore.case = TRUE))
}

is_path <- function(path) {
    if (is_directory(path)) {
        out <- TRUE
    } else if (is_csv_file(path)) {
        out <- FALSE
    } else {
        stop('Input argument must be either a path to individual data files or an individual data file.')
    }
    return(out)
}

# If a whole path is supplied
if (is_path(path)) {
    d_list <- import_dlist(path)
    d_list <- lapply(d_list, assign_response)
    d <- do.call(rbind, d_list)
    
# If only one csv is supplied
} else {
    d <- assign_response(read.csv(path))
}

infer_path()

# Save output
if (!dir.exists('formatted_data')) {
    dir.create('formatted_data')
}

datetime <- format(Sys.time(), '%Y%m%d%H%M%S')
write.csv(d, paste0('formatted_data/GradCPT_', datetime, '.csv'), row.names = FALSE)


