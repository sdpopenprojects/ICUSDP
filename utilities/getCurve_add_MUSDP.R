library("ScottKnottESD")
library("reshape2")
library("car")
library("effsize")
source("E:/ICUSDP/INTC/ICUSDP/utilities/my_ESD.R")

#### plot methods #####

AR <- NULL
cols <- c("black","red", "blue", "purple", "seagreen", "salmon", "orange",
          "brown", "skyblue", "orchid", "sienna", "pink", "gold", "green",
          "cyan", "plum","lightblue", "tan", "gray")

# 扩展为 MUSDP 具备的 13 个指标（与生成的数据文件名保持一致）
meas <- c(
  'AUC', 'g_mean', 'precision', 'recall', 'pf',
  'F1', 'MCC', 'Popt', 'cErecall', 'cEprecision',
  'cEfmeasure', 'cPMI', 'cIFA'
)

# 画图和绘图标签显示的指标名称
meas2 <- c(
  'AUC', 'G_mean', 'Precision', 'Recall', 'PF',
  'F1', 'MCC', 'Popt', 'Recall@20%LOC', 'Precision@20%LOC',
  'F_measure@20%', 'PMI@20%LOC', 'IFA@20%LOC'
)

projectNames <- c(
  'activemq-5.0.0', 'activemq-5.1.0', 'activemq-5.2.0', 'activemq-5.3.0', 'activemq-5.8.0',
  'derby-10.2.1.6', 'derby-10.3.1.4', 'derby-10.5.1.1', 'groovy-1_5_7' , 'groovy-1_6_BETA_1' ,
  'groovy-1_6_BETA_2' ,'hbase-0.94.0' ,'hbase-0.95.0' ,'hbase-0.95.2' ,'hive-0.9.0' ,'hive-0.10.0',
  'hive-0.12.0', 'jruby-1.1' ,'jruby-1.4.0', 'jruby-1.5.0' ,'jruby-1.7.0.preview1' ,'lucene-2.3.0',
  'lucene-2.9.0', 'lucene-3.0.0', 'lucene-3.1', 'wicket-1.3.0-beta2', 'wicket-1.3.0-incubating-beta-1', 
  'wicket-1.5.3'
)

# 输入目录
input_path <- "E:/ICUSDP/INTC/ICUSDP/W_C_test/visual/SKESD/dataall_unsupervised1/"

# 输出目录
result_dir <- "E:/ICUSDP/INTC/ICUSDP/W_C_test/visual/SKESD/USDP1/"

# 创建输出目录
dir.create(result_dir, recursive = TRUE, showWarnings = FALSE)

# 创建图片目录
plot_dir <- paste0(result_dir, "plots/")
dir.create(plot_dir, recursive = TRUE, showWarnings = FALSE)


for (meas_idx in seq(meas)) {
  mea <- meas[meas_idx]
  mea2 <- meas2[meas_idx]
  
  save_path <- paste0(result_dir, "NPSKESD_group_", mea, "_CLF.csv")
  data_file <- paste0(input_path, "all_results_", mea, ".csv")
  
  data <- read.csv(data_file, 
                   header = FALSE, 
                   nrows = 2800)
  
  # 包含 MUSDP 的 10 种对比方法名称
  mnames2 <- c('ICUSDP', 'MUSDP', 'MU', 'MD', 'SC', 'TCL', 'TCLP', 'CLA', 'CLAMI', 'ONE')
  colnames(data) <- mnames2
  
  numberOfProjects <- 28
  len <- nrow(data) / numberOfProjects
  sk1st <- NULL
  
  pdf_file <- paste0(plot_dir, "NPSKESD_CLF_", mea, ".pdf")
  pdf(file = pdf_file, width = 12, height = 66, paper = "special")
  
  opar <- par(no.readonly = TRUE)
  par(mfrow=c(numberOfProjects,1))
  
  for (i in seq(numberOfProjects)) {
    idx1 <- (i - 1) * len + 1
    idx2 <- i * len
    curdata <- data[idx1:idx2 , ]
    
    # 对越小越好的指标取负值进行正向转换
    if (mea %in% c("pf", "cPMI", "cPCI", "mPCI", "mPMI", "cIFA", "mIFA", "ceIFA", "meIFA")){
      sk <- my_sk_esd(-curdata, version="np")
      sk$m.inf <- abs(sk$m.inf)
    } else{
      sk <- my_sk_esd(curdata, version="np")
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