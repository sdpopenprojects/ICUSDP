library("ScottKnottESD")
library("reshape2")
library("car")
library("effsize")

# 1. 绝对路径引入 my_ESD.R
source("F:/ICUSDP/INTC/ICUSDP/utilities/my_ESD.R")

#### 核心配置 #####

# 最终生成的全指标综合排名矩阵（每一行代表一个指标，每一列代表一个模型）
FINAL_AR_MATRIX <- NULL

cols <- c("black","red", "blue", "purple", "seagreen", "salmon", "orange",
          "brown", "skyblue",  "orchid", "sienna", "pink", "gold", "green",
          "cyan",  "plum","lightblue", "tan", "gray")

# 16 个模型的标准排列顺序
mnames2 <- c('ICUSDP', 'KMedoids', 'MU', 'MD', 'SC', 'TCL', 
             'TCLP', 'CLA', 'CLAMI', 'ONE', 'DT', 'RF', 'GBM', 'XGBoost', 'LR', 'linearSVM')

data_type <- "O"

# 输入路径与输出路径
input_dir <- "F:/ICUSDP/INTC/ICUSDP/W_C_test/statistic/all_before/"
base_path <- "F:/ICUSDP/INTC/ICUSDP/W_C_test/visual/visual_result/"

# 32 个标准的指标名称
meas <- c(
  'precision', 'recall', 'pf', 'F1', 'AUC', 'g_measure', 'g_mean', 'bal', 'MCC', 'accuracy',
  'Popt', 'cErecall', 'cEprecision', 'cEfmeasure', 'cMCC', 'cPMI', 'cIFA', 'cPCI', 'c_ROI_PII', 'c_ROI_PCI', 'ceIFA',
  'mRecall', 'mPrecision', 'mfmeasure', 'mMCC', 'mPMI', 'mIFA', 'mPCI', 'm_ROI_PII', 'm_ROI_PCI', 'meIFA',
  'time' 
)

cat("🔍 正在扫描输入目录下的中位数汇总 CSV 文件...\n")
all_existing_files <- list.files(input_dir, pattern = "\\.csv$", full.names = TRUE, ignore.case = TRUE)

if (length(all_existing_files) == 0) {
  stop(paste("❌ 严重错误：在输入目录", input_dir, "下没有找到任何 CSV 文件！"))
}

# ================= 🔄 核心全局检验循环（28行数据集维度） =================

for (meas_idx in seq(meas)) {
  mea <- meas[meas_idx]
  
  result_dir <- paste0(base_path, data_type, "_results/")
  dir.create(result_dir, showWarnings = FALSE, recursive = TRUE)
  
  save_path <- paste0(result_dir, "NPSKESD_group_", mea, "_CLF.csv")
  if (file.exists(save_path)) { file.remove(save_path) }
  
  # ------ 🔧 横向横跨 16 个模型提取当前指标的 28 行项目数据 ------
  data <- NULL
  
  for (model in mnames2) {
    # 模糊精准匹配包含模型名字的 28 行汇总文件
    matched_file <- all_existing_files[grep(model, basename(all_existing_files), ignore.case = TRUE)]
    
    if (length(matched_file) == 0) {
      stop(paste0("❌ 错误：找不到模型 [", model, "] 对应的汇总 CSV 文件！"))
    } else if (length(matched_file) > 1) {
      matched_file <- matched_file[1]
    }
    
    # 28行数据带表头，第一列为项目名
    raw_data <- read.csv(matched_file, header = TRUE, row.names = 1)
    
    # 对 time 指标在不同文件中的可能列名做智能兼容映射
    if ("median_time" %in% colnames(raw_data)) {
      colnames(raw_data)[colnames(raw_data) == "median_time"] <- "time"
    } else if ("mean_time" %in% colnames(raw_data)) {
      colnames(raw_data)[colnames(raw_data) == "mean_time"] <- "time"
    }
    
    # 抓取当前循环对应的指标列
    col_idx <- which(colnames(raw_data) == mea)
    if (length(col_idx) == 0) {
      stop(paste0("❌ 错误：在文件 [", basename(matched_file), "] 中找不到指标 [", mea, "]！请核对列名。"))
    }
    
    data <- cbind(data, raw_data[, col_idx])
  }
  
  data <- as.data.frame(data)
  colnames(data) <- mnames2 # 组装成完美的 28行(数据集) × 16列(模型) 矩阵
  
  # ------ 💥 跨数据集全局 Scott-Knott ESD 检验 ------
  # 自动识别越小越好的指标并取反
  if (tolower(mea) %in% c("pf", "pmi", "ifa", "ceifa", "meifa", "time")) {
    sk <- my_sk_esd(-data, version="np")
    sk$m.inf <- abs(sk$m.inf)
  } else {
    sk <- my_sk_esd(data, version="np")
  }
  
  # 精准按 mnames2 名字顺序强行恢复并提取 Rank 结果
  current_groups <- sk$groups
  ordered_groups <- current_groups[mnames2]
  names(ordered_groups) <- mnames2
  
  # 导出单个指标的 Rank 结果
  write.table(t(ordered_groups), file = save_path, sep=',',
              append = FALSE, row.names = FALSE, col.names = TRUE)
  
  # 纵向追加至最终综合矩阵
  FINAL_AR_MATRIX <- rbind(FINAL_AR_MATRIX, ordered_groups)
  
  # ------ 📊 绘制单指标全局汇总图（用于论文插图） ------
  plot_dir <- paste0(result_dir, "plots/")
  dir.create(plot_dir, showWarnings = FALSE)
  
  summary_pdf <- paste0(plot_dir, "NPSKESD_CLF_", mea, "_all.pdf")
  pdf(file = summary_pdf, width = 12, height = 4, paper = "special")
  par(mar = c(6, 4, 2, 2))
  
  plot(sk, title = "", xlab = "", ylab = "Rank", col = cols, las = 2)
  title(main = paste("Global Dataset-level S-K ESD Rank for:", mea), cex = 0.9)
  
  dev.off()
}

# ================= 💾 保存最终无全1的综合矩阵 =================
FINAL_AR_MATRIX <- as.matrix(FINAL_AR_MATRIX)
rownames(FINAL_AR_MATRIX) <- meas
colnames(FINAL_AR_MATRIX) <- mnames2

write.csv(FINAL_AR_MATRIX, file = paste0(base_path, data_type, "_results/AR_CLF.csv"))

cat("\n🎉 [大功告成] 28行跨数据集全局检验已全部成功运行完毕！")
cat("\n📂 请立即去查看全新的 AR_CLF.csv，错位和全1魔咒已彻底解除！\n")