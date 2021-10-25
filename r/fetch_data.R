setwd('./projectx-2021')
library(TCGAbiolinks)
library(foreach)
library(magrittr)
library(tibble)
library(readr)
projects <- TCGAbiolinks:::getGDCprojects()$project_id
projects <- projects[grepl('^TCGA', projects, perl = TRUE)]
foreach(project = projects) %do% {
  project.dir <- file.path(".", "data", project)
  dir.create(project.dir, recursive = TRUE)
  project.summary <-
    TCGAbiolinks:::getProjectSummary(project) %>% .$data_categories %>% as_tibble
  project.summary %>% write_csv(paste(project.dir,
                                      paste("project.summary",
                                            "csv",
                                            sep = "."),
                                      sep = "/"))
  query.gene.expr <- GDCquery(project = project,
                              data.category = "Transcriptome Profiling",
                              data.type = "Gene Expression Quantification")
  gene.expr <-
    query.gene.expr %>% getResults %>% as_tibble
  gene.expr %>% write_csv(paste(project.dir,
                                paste("gene.expr",
                                      "csv",
                                      sep = "."),
                                sep = "/"))
  tryCatch(
    GDCdownload(
      query.gene.expr,
      method = "api",
      files.per.chunk = 20
    ),
    error = function(e)
      GDCdownload(query.gene.expr,
                  method = "client")
  )
}