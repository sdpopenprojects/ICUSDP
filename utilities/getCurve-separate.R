library("ScottKnottESD")
library("reshape2")
library("car")
library("effsize")

# 1. 绝对路径引入 my_ESD.R
source("F:/ICUSDP/INTC/ICUSDP/utilities/my_ESD.R")

#### 🎨 绘图颜色配置 #####
cols <- c("black","red", "blue", "purple", "seagreen", "salmon", "orange",
          "brown", "skyblue",  "orchid", "sienna", "pink", "gold", "green",
          "cyan",  "plum","lightblue", "tan", "gray")

# 32 个标准的指标名称
meas <- c(
  'precision', 'recall', 'pf', 'F1', 'AUC', 'g_measure', 'g_mean', 'bal', 'MCC', 'accuracy',
  'Popt', 'cErecall', 'cEprecision', 'cEfmeasure', 'cMCC', 'cPMI', 'cIFA', 'cPCI', 'c_ROI_PII', 'c_ROI_PCI', 'ceIFA',
  'mRecall', 'mPrecision', 'mfmeasure', 'mMCC', 'mPMI', 'mIFA', 'mPCI', 'm_ROI_PII', 'm_ROI_PCI', 'meIFA',
  'time' 
)

# 📂 统一输出的基准总路径
base_output_path <- "F:/ICUSDP/INTC/ICUSDP/W_C_test/visual/visual_result/SDP VS USDP/"
dir.create(base_output_path, showWarnings = FALSE, recursive = TRUE)

# ==================== 🛠️ 定义两组实验的对比阵营与输入路径 ====================
experiment_modes <- list(
  Supervised = list(
    input_dir = "F:/ICUSDP/INTC/ICUSDP/W_C_test/visual/data/supervised/",
    mnames = c('ICUSDP', 'DT', 'RF', 'GBM', 'XGBoost', 'LR', 'linearSVM')
  ),
  Unsupervised = list(
    input_dir = "F:/ICUSDP/INTC/ICUSDP/W_C_test/visual/data/unsupervised/",
    mnames = c('ICUSDP', 'KMedoids', 'MU', 'MD', 'SC', 'TCL', 'TCLP', 'CLA', 'CLAMI', 'ONE')
  )
)

# ==================== 🔄 开始分组实验大循环 ====================

for (mode_name in names(experiment_modes)) {
  
  mode_cfg <- experiment_modes[[mode_name]]
  current_input_dir <- mode_cfg$input_dir
  current_models <- mode_cfg$mnames
  
  cat(paste0("\n🚀 [正在启动] 正在执行针对 <<< ", mode_name, " >>> 阵营的对比检验...\n"))
  cat(paste0("📂 输入路径: ", current_input_dir, "\n"))
  
  # 获取当前目录下的所有 CSV
  all_existing_files <- list.files(current_input_dir, pattern = "\\.csv$", full.names = TRUE, ignore.case = TRUE)
  
  # 统一创建本组的输出子目录
  result_dir <- paste0(base_output_path, mode_name, "_results/")
  dir.create(result_dir, showWarnings = FALSE, recursive = TRUE)
  
  # 创建图片保存子目录
  plot_dir <- paste0(result_dir, "plots/")
  dir.create(plot_dir, showWarnings = FALSE)
  
  # 初始化当前组的最终 AR 排名矩阵
  FINAL_AR_MATRIX <- NULL
  
  # ----------- 🔄 遍历 32 个指标 -----------
  for (meas_idx in seq(meas)) {
    mea <- meas[meas_idx]
    
    # 建立单个指标的局部组别 CSV 路径
    save_path <- paste0(result_dir, "NPSKESD_group_", mea, "_CLF.csv")
    if (file.exists(save_path)) { file.remove(save_path) }
    
    data <- NULL
    
    # ----------- 🔧 纵向横跨当前阵营的模型抽取数据 -----------
    for (model in current_models) {
      
      # 在当前组输入文件夹下匹配该模型文件
      matched_file <- all_existing_files[grep(model, basename(all_existing_files), ignore.case = TRUE)]
      
      # 🌟 特殊安全防护：如果跑监督组时，监督目录下没有 ICUSDP.csv，则去无监督目录下调取
      if (length(matched_file) == 0 && model == "ICUSDP") {
        backup_dir <- experiment_modes$Unsupervised$input_dir
        backup_files <- list.files(backup_dir, pattern = "\\.csv$", full.names = TRUE, ignore.case = TRUE)
        matched_file <- backup_files[grep(model, basename(backup_files), ignore.case = TRUE)]
      }
      
      if (length(matched_file) == 0) {
        stop(paste0("❌ 错误：在组别 [", mode_name, "] 中找不到模型 [", model, "] 对应的汇总 CSV 文件！"))
      } else if (length(matched_file) > 1) {
        matched_file <- matched_file[1]
      }
      
      # 读取 28 行带表头的中位数汇总数据
      raw_data <- read.csv(matched_file, header = TRUE, row.names = 1)
      
      # 智能列名映射（处理 time）
      if ("median_time" %in% colnames(raw_data)) {
        colnames(raw_data)[colnames(raw_data) == "median_time"] <- "time"
      } else if ("mean_time" %in% colnames(raw_data)) {
        colnames(raw_data)[colnames(raw_data) == "mean_time"] <- "time"
      }
      
      # 抽取指标列
      col_idx <- which(colnames(raw_data) == mea)
      if (length(col_idx) == 0) {
        stop(paste0("❌ 错误：在文件 [", basename(matched_file), "] 中找不到指标 [", mea, "]！"))
      }
      
      data <- cbind(data, raw_data[, col_idx])
    }
    
    data <- as.data.frame(data)
    colnames(data) <- current_models  # 完美组装本组当前指标的 28行 × 模型数 矩阵
    
    # ----------- 💥 执行 Scott-Knott ESD 全局检验 -----------
    # 自动识别越小越好的指标并取反
    if (tolower(mea) %in% c("pf", "pmi", "ifa", "ceifa", "meifa", "time")) {
      sk <- my_sk_esd(-data, version="np")
      sk$m.inf <- abs(sk$m.inf)
    } else {
      sk <- my_sk_esd(data, version="np")
    }
    
    # 按当前组模型定义的标准顺序对齐并提取 Rank
    current_groups <- sk$groups
    ordered_groups <- current_groups[current_models]
    names(ordered_groups) <- current_models
    
    # 导出当前指标的单文件 Rank 结果
    write.table(t(ordered_groups), file = save_path, sep=',',
                append = FALSE, row.names = FALSE, col.names = TRUE)
    
    # 纵向追加至当前阵营的总 AR 矩阵
    FINAL_AR_MATRIX <- rbind(FINAL_AR_MATRIX, ordered_groups)
    
    # ----------- 📊 绘制并保存漂亮的 PDF 插图 -----------
    summary_pdf <- paste0(plot_dir, "NPSKESD_CLF_", mea, "_all.pdf")
    pdf(file = summary_pdf, width = 10, height = 4, paper = "special")
    par(mar = c(6, 4, 2, 2))
    
    plot(sk, title = "", xlab = "", ylab = "Rank", col = cols, las = 2)
    title(main = paste0("[", mode_name, " VS ICUSDP] S-K ESD Rank for: ", mea), cex = 0.9)
    
    dev.off()
  }
  
  # ----------- 💾 组装并保存当前阵营的总 AR_CLF.csv 矩阵 -----------
  FINAL_AR_MATRIX <- as.matrix(FINAL_AR_MATRIX)
  rownames(FINAL_AR_MATRIX) <- meas
  colnames(FINAL_AR_MATRIX) <- current_models
  
  write.csv(FINAL_AR_MATRIX, file = paste0(result_dir, "AR_CLF.csv"))
  cat(paste0("✨ [完成] ", mode_name, " 阵营的综合矩阵 AR_CLF.csv 已成功生成！\n"))
}

cat("\n🎉🎉🎉 [大功告成] 监督与无监督两组对照实验均已完美运行完毕！")
cat("\n📂 所有成果均已存入新目录：F:/ICUSDP/INTC/ICUSDP/W_C_test/visual/visual_result/SDP VS USDP/\n")