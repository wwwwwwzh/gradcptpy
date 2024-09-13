
import_dlist <- function(path) {
    files <- list.files(path=path, pattern = '.*task-gradcpt.*', full.names = TRUE)
    if (length(files) == 0) {
        stop('Input argument as a path must point to the path containing the individual GradCPT data files.')
    }
    d_list <- lapply(files, read.csv)
    return(d_list)
}

