library("ScottKnottESD")
library("reshape2")
library("car")
library("effsize")
#source("my_ESD.R")
source("F:/ICUSDP/INTC/ICUSDP/utilities/my_ESD.R")

# sk <- sk_esd(example)
# plot(sk)
#
# sk <- sk_esd(maven)
# plot(sk)

# xnames <- colnames(data)
# mdata <- apply(data[, xnames], MARGIN=2, FUN=mean)
# mdata <- data.frame(matrix(mdata, nrow=1, ncol=length(xnames)))
# colnames(mdata) <- xnames


# install.packages("devtools")
# devtools::install_github("klainfo/ScottKnottESD", ref="development")


#### plot methods #####

AR <- NULL
cols <- c("black","red", "blue", "purple", "seagreen", "salmon", "orange",
          "brown", "skyblue",  "orchid", "sienna", "pink", "gold", "green",
          "cyan",  "plum","lightblue", "tan", "gray")

#meas <- c("AUC", "MCC", "Efmeasure", "IFA")
#meas2 <- c("AUC", "MCC", "F_measure@20%", "IFA")
# 32 个无表头标准列名
meas <- c(
  'precision', 'recall', 'pf', 'F1', 'AUC', 'g_measure', 'g_mean', 'bal', 'MCC', 'accuracy',
  'Popt', 'cErecall', 'cEprecision', 'cEfmeasure', 'cMCC', 'cPMI', 'cIFA', 'cPCI', 'c_ROI_PII', 'c_ROI_PCI', 'ceIFA',
  'mRecall', 'mPrecision', 'mfmeasure', 'mMCC', 'mPMI', 'mIFA', 'mPCI', 'm_ROI_PII', 'm_ROI_PCI', 'meIFA',
  'time' 
)
meas2 <- c(
  'Precision', 'Recall', 'PF', 'F1', 'AUC', 'G_measure', 'G_mean', 'Bal', 'MCC', 'Accuracy',
  'Popt', 'Recall@20%LOC', 'Precision@20%LOC', 'F_measure@20%', 'MCC@20%LOC', 'PMI@20%LOC', 'IFA@20%LOC', 'PCI@20%LOC', 'ROI_PII@20%LOC', 'ROI_PCI@20%LOC', 'eIFA@20%LOC',
  'Recall@20%Modules', 'Precision@20%Modules', 'F_measure@20%Modules', 'MCC@20%Modules', 'PMI@20%Modules', 'IFA@20%Modules', 'PCI@20%Modules', 'ROI_PII@20%Modules', 'ROI_PCI@20%Modules', 'eIFA@20%Modules',
  'time' 
)

projectNames <- c(
  'activemq-5.0.0', 'activemq-5.1.0', 'activemq-5.2.0', 'activemq-5.3.0', 'activemq-5.8.0',
  'derby-10.2.1.6', 'derby-10.3.1.4', 'derby-10.5.1.1', 'groovy-1_5_7' , 'groovy-1_6_BETA_1' ,
  'groovy-1_6_BETA_2' ,'hbase-0.94.0' ,'hbase-0.95.0' ,'hbase-0.95.2' ,'hive-0.9.0' ,'hive-0.10.0',
  'hive-0.12.0', 'jruby-1.1' ,'jruby-1.4.0', 'jruby-1.5.0' ,'jruby-1.7.0.preview1' ,'lucene-2.3.0',
  'lucene-2.9.0', 'lucene-3.0.0', 'lucene-3.1', 'wicket-1.3.0-beta2', 'wicket-1.3.0-incubating-beta-1', 
  'wicket-1.5.3'
)

# 获取参数
# 'O', 'log', 'Z-score', 'Max-Min', 'yeo-johnson', 'rank-transformation'
#data_type <- "O"

#base_path <- "D:/ST/pycharm/Pycharm_project/TransClusterDefectX/result/all_result/test3/picture4_Last91/"
#filePath <- paste0(base_path, data_type, "/")

# 输入目录
#input_path <- "F:/ICUSDP/INTC/ICUSDP/W_C_test/visual/dataall/"
#input_path <- "F:/ICUSDP/INTC/ICUSDP/W_C_test/visual/SKE/dataall_unsupervised/"
input_path <- "F:/ICUSDP/INTC/ICUSDP/W_C_test/visual/SKE/dataall_supervised/"

# 输出目录
#result_dir <- "F:/ICUSDP/INTC/ICUSDP/W_C_test/visual/visual_result/SDP VS USDP/"
#result_dir <- "F:/ICUSDP/INTC/ICUSDP/W_C_test/visual/SKE/USDP/"
result_dir <- "F:/ICUSDP/INTC/ICUSDP/W_C_test/visual/SKE/SDP/"

# 创建输出目录
dir.create(result_dir, recursive = TRUE, showWarnings = FALSE)

# 创建图片目录
plot_dir <- paste0(result_dir, "plots/")
dir.create(plot_dir, recursive = TRUE, showWarnings = FALSE)


for (meas_idx in seq(meas)) {
  mea <- meas[meas_idx]
  mea2 <- meas2[meas_idx]
  
  # 创建结果目录（如果不存在）
  #result_dir <- paste0(base_path, data_type, "_results/")
  #dir.create(result_dir, showWarnings = FALSE)
  
  #save_path <- paste0(result_dir, "NPSKESD_group_", mea, "_CLF.csv")
  #data_file <- paste0(filePath, mea, "/all_results_", mea, ".csv")
  
  save_path <- paste0(result_dir, "NPSKESD_group_", mea, "_CLF.csv")
  data_file <- paste0(input_path, "all_results_", mea, ".csv")
  
  data <- read.csv(data_file, 
                   header = FALSE, 
                   nrows = 2800)
 
  #mnames <- c('ICUSDP', 'KMedoids', 'MU', 'MD', 'SC', 'TCL', 'TCLP', 'CLA', 'CLAMI', 'ONE', 'DT', 'RF', 'GBM', 'XGBoost', 'LR', 'linearSVM')
  #mnames <- c('ICUSDP', 'KMedoids', 'MU', 'MD', 'SC', 'TCL', 'TCLP', 'CLA', 'CLAMI', 'ONE')
  mnames <- c('ICUSDP', 'DT', 'RF', 'GBM', 'XGBoost', 'LR', 'linearSVM')
  
  
  #mnames2 <- c('ICUSDP', 'KMedoids', 'MU', 'MD', 'SC', 'TCL', 'TCLP', 'CLA', 'CLAMI', 'ONE', 'DT', 'RF', 'GBM', 'XGBoost', 'LR', 'linearSVM')
  #mnames2 <- c('ICUSDP', 'KMedoids', 'MU', 'MD', 'SC', 'TCL', 'TCLP', 'CLA', 'CLAMI', 'ONE')
  mnames2 <- c('ICUSDP', 'DT', 'RF', 'GBM', 'XGBoost', 'LR', 'linearSVM')
  
  
  
  
  colnames(data) <- mnames2
  
  numberOfProjects <- 28
  len <- nrow(data) / numberOfProjects
  sk1st <- NULL
  
  # 创建绘图目录
  #plot_dir <- paste0(result_dir, "plots/")
  #dir.create(plot_dir, showWarnings = FALSE)
  
  pdf_file <- paste0(plot_dir, "NPSKESD_CLF_", mea, ".pdf")
  pdf(file = pdf_file, width = 12, height = 66, paper = "special")
  
  opar <- par(no.readonly = TRUE)
  par(mfrow=c(numberOfProjects,1))
  
  for (i in seq(numberOfProjects)) {
    idx1 <- (i - 1) * len + 1
    idx2 <- i * len
    curdata <- data[idx1:idx2 , ]
    
    if (mea %in% c("pf", "cPMI", "cPCI", "mPCI", "mPMI", "cIFA", "mIFA", "ceIFA", "meIFA")){
      sk <- my_sk_esd(-curdata, version="np")
      #sk <- sk_esd(-curdata, version="np")
      sk$m.inf <- abs(sk$m.inf)
    } else{
      sk <- my_sk_esd(curdata, version="np")
      #sk <- sk_esd(curdata, version="np")
    }
    
    plot(sk, title = "", xlab = "", ylab = mea2, col=cols, las=2)
    title(main = projectNames[i], cex = 0.8)
    
    write.table(t(sk$groups), file = save_path, sep=',',
                append = TRUE, row.names= FALSE, col.names = TRUE)
    
    sk1st <- rbind(sk1st, sk$groups[order(sk$ord)])
  }
  
  par(opar)
  dev.off()
  
  ar <- colMeans(sk1st)
  AR <- rbind(AR, ar)
  
  # sk <- my_sk_esd(sk1st, version="np")
  sk <- sk_esd(sk1st, version="np")
  
  summary_pdf <- paste0(plot_dir, "NPSKESD_CLF_", mea, "_all.pdf")
  pdf(file = summary_pdf, width = 12, height = 3, paper = "special")
  par(mar = c(6, 4, 2, 0))
  plot(sk, title = "", xlab = "", ylab = "Rank", col=cols, las=2)
  title(main = mea2, cex = 0.8)
  
  write.table(t(sk$groups), file = save_path, sep=',',
              append = TRUE, row.names= FALSE, col.names = TRUE)
  
  dev.off()
}

rownames(AR) <- meas
write.csv(AR, file = paste0(result_dir, "AR_CLF.csv"))
print(AR)
