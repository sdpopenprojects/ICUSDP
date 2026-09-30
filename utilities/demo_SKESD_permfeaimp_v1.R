library("ScottKnottESD")
# library("reshape2")
# library("car")
# library("effsize")


# ============================================================
# Plot methods
# ============================================================

AR <- NULL

cols <- c(
  "black", "red", "blue", "purple", "seagreen", "salmon",
  "orange", "brown", "skyblue", "orchid", "sienna", "pink",
  "gold", "green", "cyan", "plum", "lightblue", "tan", "gray"
)


# ============================================================
# 1. Input path
# ============================================================

path <- "E:/ICUSDP/INTC/ICUSDP-main/new/noFS/pkl_results_individual"


# ============================================================
# 2. Get all CSV files
# ============================================================

files <- list.files(
  path = path,
  pattern = "\\.csv$",
  full.names = TRUE
)

files <- sort(files)

n <- length(files)

cat("========================================\n")
cat("Input directory:\n")
cat(path, "\n")
cat("Number of CSV files:", n, "\n")
cat("========================================\n")


if (n == 0) {
  stop(
    "No CSV files were found in the input directory:\n",
    path
  )
}


# ============================================================
# 3. Output path
# ============================================================

path2 <- "E:/ICUSDP/INTC/ICUSDP-main/new/noFS/TOPK"


# Create output directory
dir.create(
  path2,
  recursive = TRUE,
  showWarnings = FALSE
)


# Check output directory
if (!dir.exists(path2)) {
  stop(
    "Output directory does not exist and could not be created:\n",
    path2
  )
}


# Check write permission
if (file.access(path2, 2) != 0) {
  stop(
    "The output directory is not writable:\n",
    path2,
    "\nPlease check whether the directory is open or protected."
  )
}


# ============================================================
# 4. Delete old output files
# ============================================================
#
# The original code uses append = TRUE.
# Therefore, old results must be removed before a new run.
#

old_files <- c(
  "npskesd_topk_v1.csv",
  "top1_v1.csv",
  "top2_v1.csv",
  "top3_v1.csv",
  "npskesd_v1.csv",
  "npskesd_AR_v1.csv"
)

for (f in file.path(path2, old_files)) {
  
  if (file.exists(f)) {
    
    result <- file.remove(f)
    
    if (result) {
      cat("Deleted old file:", basename(f), "\n")
    } else {
      warning(
        "Could not delete old file: ",
        f,
        "\nPlease make sure the file is not opened by Excel/WPS."
      )
    }
    
  }
}


# ============================================================
# 5. Output file
# ============================================================

x <- 1

save_path <- file.path(
  path2,
  paste0("npskesd_v", x, ".csv")
)


# ============================================================
# 6. Unified 37 features
# ============================================================

allfeas <- c(
  "Added_lines",
  "ADEV",
  "AvgCyclomatic",
  "AvgCyclomaticModified",
  "AvgCyclomaticStrict",
  "AvgEssential",
  "AvgLine",
  "AvgLineBlank",
  "AvgLineCode",
  "AvgLineComment",
  "COMM",
  "CountClassBase",
  "CountClassCoupled",
  "CountClassDerived",
  "CountDeclClass",
  "CountDeclClassMethod",
  "CountDeclClassVariable",
  "CountDeclFunction",
  "CountDeclInstanceMethod",
  "CountDeclInstanceVariable",
  "CountDeclMethod",
  "CountDeclMethodDefault",
  "CountDeclMethodPrivate",
  "CountDeclMethodProtected",
  "CountDeclMethodPublic",
  "CountInput_Mean",
  "CountInput_Min",
  "CountInput_Max",
  "CountLine",
  "CountLineBlank",
  "CountLineCode",
  "CountLineCodeDecl",
  "CountLineCodeExe",
  "CountLineComment",
  "CountOutput_Mean",
  "CountOutput_Min",
  "CountOutput_Max",
  "CountPath_Mean",
  "CountPath_Min",
  "CountPath_Max",
  "CountSemicolon",
  "CountStmt",
  "CountStmtDecl",
  "CountStmtExe",
  "DDEV",
  "Del_lines",
  "MAJOR_COMMIT",
  "MAJOR_LINE",
  "MaxCyclomatic",
  "MaxCyclomaticModified",
  "MaxCyclomaticStrict",
  "MaxInheritanceTree",
  "MaxNesting_Mean",
  "MaxNesting_Min",
  "MaxNesting_Max",
  "MINOR_COMMIT",
  "MINOR_LINE",
  "OWN_COMMIT",
  "OWN_LINE",
  "PercentLackOfCohesion",
  "RatioCommentToCode",
  "SumCyclomatic",
  "SumCyclomaticModified",
  "SumCyclomaticStrict",
  "SumEssential"
)


# Check number of features
cat("\nNumber of unified features:", length(allfeas), "\n")

#if (length(allfeas) != 37) {
  #stop(
  #  "ERROR: allfeas should contain exactly 37 features, ",
  #  "but currently contains ",
  #  length(allfeas),
  #  "."
 # )
#}


# ============================================================
# 7. Create zero matrix
# ============================================================
#
# 100 repetitions × 37 unified features
#
# For each project:
# - existing features keep their original values
# - missing features are supplemented with 0
#

tdata <- matrix(
  0,
  nrow = 100,
  ncol = length(allfeas)
)

colnames(tdata) <- allfeas


# ============================================================
# 8. Initialize storage
# ============================================================

sk1st <- NULL


# ============================================================
# 9. Process every project CSV
# ============================================================

for (i in 1:n) {
  
  cat("\n")
  cat("----------------------------------------\n")
  cat("Processing:", i, "/", n, "\n")
  cat("File:", basename(files[i]), "\n")
  cat("----------------------------------------\n")
  
  
  # ==========================================================
  # 9.1 Read CSV
  # ==========================================================
  
  data <- read.csv(
    file = files[i],
    header = TRUE,
    sep = ","
  )
  
  
  data <- data[, colnames(data) != "Round", drop = FALSE]
  
  
  # ==========================================================
  # 9.2 Check number of repetitions
  # ==========================================================
  
  cat(
    "Rows:",
    nrow(data),
    " Columns:",
    ncol(data),
    "\n"
  )
  
  if (nrow(data) != 100) {
    
    warning(
      basename(files[i]),
      " contains ",
      nrow(data),
      " rows instead of 100."
    )
    
  }
  
  
  # ==========================================================
  # 9.3 Align to the unified 37-feature space
  # ==========================================================
  #
  # IMPORTANT:
  # This is the original logic and is intentionally preserved.
  #
  # Existing features:
  #     keep original values
  #
  # Missing features:
  #     filled with 0
  #
  # duplicated feature names:
  #     keep the first occurrence
  #
  
  df <- cbind(
    data,
    tdata
  )
  
  df <- df[
    ,
    !duplicated(colnames(df))
  ]
  
  
  # ==========================================================
  # 9.4 Check the aligned data
  # ==========================================================
  
  cat(
    "Columns after alignment:",
    ncol(df),
    "\n"
  )
  
  
  # ==========================================================
  # 9.5 Scott-Knott ESD
  # ==========================================================
  
  sk <- sk_esd(
    df,
    version = "np"
  )
  
  
  # ==========================================================
  # 9.6 Save top 3 ranks
  # ==========================================================
  
  feaList <- NULL
  
  k <- 3
  
  
  for (j in 1:k) {
    
    fea <- names(
      sk$groups[
        sk$groups == j
      ]
    )
    
    fea <- paste(
      fea,
      collapse = ','
    )
    
    feaList <- c(
      feaList,
      fea
    )
    
    
    # --------------------------------------------------------
    # Top-1
    # --------------------------------------------------------
    
    if (j == 1) {
      
      top1 <- top2 <- top3 <- fea
      
    }
    
    
    # --------------------------------------------------------
    # Top-2
    # --------------------------------------------------------
    
    if (j == 2) {
      
      top2 <- paste(
        top2,
        fea,
        sep = ','
      )
      
      top3 <- paste(
        top3,
        fea,
        sep = ','
      )
      
    }
    
    
    # --------------------------------------------------------
    # Top-3
    # --------------------------------------------------------
    
    if (j == 3) {
      
      top3 <- paste(
        top3,
        fea,
        sep = ','
      )
      
    }
    
  }
  
  
  # ==========================================================
  # 9.7 Save top-k feature groups
  # ==========================================================
  
  write.table(
    t(feaList),
    file = file.path(
      path2,
      "npskesd_topk_v1.csv"
    ),
    sep = ",",
    append = TRUE,
    row.names = FALSE,
    col.names = FALSE
  )
  
  
  # ==========================================================
  # 9.8 Save top-1
  # ==========================================================
  
  write.table(
    t(top1),
    file = file.path(
      path2,
      "top1_v1.csv"
    ),
    sep = ",",
    append = TRUE,
    row.names = FALSE,
    col.names = FALSE
  )
  
  
  # ==========================================================
  # 9.9 Save top-2
  # ==========================================================
  
  write.table(
    t(top2),
    file = file.path(
      path2,
      "top2_v1.csv"
    ),
    sep = ",",
    append = TRUE,
    row.names = FALSE,
    col.names = FALSE
  )
  
  
  # ==========================================================
  # 9.10 Save top-3
  # ==========================================================
  
  write.table(
    t(top3),
    file = file.path(
      path2,
      "top3_v1.csv"
    ),
    sep = ",",
    append = TRUE,
    row.names = FALSE,
    col.names = FALSE
  )
  
  
  # ==========================================================
  # 9.11 Save SK-ESD groups
  # ==========================================================
  
  write.table(
    t(sk$groups),
    file = save_path,
    sep = ",",
    append = TRUE,
    row.names = FALSE,
    col.names = TRUE
  )
  
  
  # ==========================================================
  # 9.12 Store groups for AR calculation
  # ==========================================================
  
  sk1st <- rbind(
    sk1st,
    sk$groups[
      order(sk$ord)
    ]
  )
  
}


# ============================================================
# 10. Calculate AR
# ============================================================

ar <- colMeans(
  sk1st
)

AR <- rbind(
  AR,
  ar
)


# ============================================================
# 11. Run SK-ESD on aggregated results
# ============================================================

sk <- sk_esd(
  sk1st,
  version = "np"
)


# ============================================================
# 12. Save transformed group results
# ============================================================

temp <- sk$groups

temp <- max(temp) - temp + min(temp)

temp <- t(
  rev(temp)
)


write.table(
  temp,
  file = save_path,
  sep = ",",
  append = TRUE,
  row.names = FALSE,
  col.names = TRUE
)


# ============================================================
# 13. Save final SK-ESD groups
# ============================================================

write.table(
  t(sk$groups),
  file = save_path,
  sep = ",",
  append = TRUE,
  row.names = FALSE,
  col.names = TRUE
)


# ============================================================
# 14. Save AR
# ============================================================

write.csv(
  AR,
  file = file.path(
    path2,
    "npskesd_AR_v1.csv"
  )
)


# ============================================================
# 15. Finished
# ============================================================

cat("\n")
cat("========================================\n")
cat("SK-ESD analysis completed successfully!\n")
cat("========================================\n")
cat("Input files       :", n, "\n")
cat("Unified features  :", length(allfeas), "\n")
cat("Output directory  :", path2, "\n")
cat("========================================\n")