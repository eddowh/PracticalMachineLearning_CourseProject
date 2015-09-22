pml_write_files = function(x){
    if (!(dir.exists('./submission_files'))) {
        dir.create('./submission_files')
    }
    n = length(x)
    for(i in 1:n){
        filename = paste0("problem_id_",i,".txt")
        write.table(x[i],
                    file=paste0("./submission_files/", filename),
                    quote=FALSE,
                    row.names=FALSE,
                    col.names=FALSE)
    }
}