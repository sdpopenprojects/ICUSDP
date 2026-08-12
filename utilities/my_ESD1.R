library("ScottKnottESD")
library("reshape2")
library("car")
library("effsize")

# 【修改 1】：中心趋势计算改为中位数 (Median) 与 IQR (Q25, Q75)
my_m.inf.1a <- function (x, which, dispersion = c("mm", "s", "se")) 
{
  # 彻底替换原本的 mean/sd 逻辑，改为计算 median, Q25, Q75
  m.inf <- aggregate(x$model[, 1], by = list(x$model[[which]]), 
                     function(x) c(median = median(x), 
                                   `Q25` = unname(quantile(x, 0.25)), 
                                   `Q75` = unname(quantile(x, 0.75))))[, 2]
  return(m.inf)
}

my_Partition <- function(g, means, mMSE, dfr, sig.level, av, k, group, ngroup, 
                         markg, g1 = g, sqsum = rep(0, g1)) {
  
  # 计算 sqsum
  for (k1 in k:(g - 1)) {
    t1 <- sum(means[k:k1])
    k2 <- g - k1
    t2 <- sum(means[(k1 + 1):g])
    sqsum[k1] <- t1^2/(k1 - k + 1) + t2^2/k2 - (t1 + t2)^2/(g - k + 1)
  }
  
  # 处理 sqsum 相同的情况
  max_sqsum <- max(sqsum)
  ord1 <- which(sqsum == max_sqsum)
  if(length(ord1) > 1) {
    ord1 <- ord1[ceiling(length(ord1)/2)]
  } else {
    ord1 <- ord1[1]
  }
  
  # 【修改 2】： diff 函数改用 Cliff's Delta (非参数效应量)
  diff <- function(k, g, av, means) {
    if (k == g) return(TRUE)
    if (means[k] == means[g]) return(TRUE)  # 保留你原本的相同值防死锁保护
    
    a <- av$model[av$model[, 2] == names(means[k]), 1]
    b <- av$model[av$model[, 2] == names(means[g]), 1]
    
    # 增加零方差与空值边界保护
    if (length(a) == 0 || length(b) == 0) return(FALSE)
    if (var(a) == 0 && var(b) == 0) return(TRUE)
    
    # 将 cohen.d 替换为非参数效应量 cliff.delta
    magnitude <- as.character(effsize::cliff.delta(a, b)$magnitude)
    return(magnitude == "negligible")
  }
  
  cond_diff <- diff(k, g, av, means)
  cond_ord1 <- (ord1 == k)
  
  if (isTRUE(cond_diff) || isTRUE(cond_ord1)) {
    if (!cond_diff) {
      ngroup <- ngroup + 1
      group[k] <- ngroup
      k <- ord1 + 1
    }
    if (cond_diff) {
      ngroup <- ngroup + 1
      group[k:g] <- ngroup
      if (prod(group) > 0) return(group)
      k <- g + 1
      g <- markg[g]
    }
    
    while (k == g) {
      ngroup <- ngroup + 1
      group[g] <- ngroup
      if (prod(group) > 0) return(group)
      k <- g + 1
      g <- markg[g]
    }
  } else {
    markg[ord1] <- g
    g <- ord1
  }
  
  my_Partition(g, means, mMSE, dfr, sig.level, av, k, group, ngroup, markg)
}

my_scottknott <- function (x, which = NULL, id.trim = 3, sig.level = 0.05, 
                           dispersion = c("mm", "s", "se"), ...) {
  if (is.null(which)) 
    which <- names(x$model)[2]
  mt <- model.tables(x, "means")
  if (is.null(mt$n)) 
    stop("No factors in the fitted model!")
  r <- mt$n[names(mt$tables)][[which]]
  MSE <- deviance(x)/df.residual(x)
  
  # 统计基础表换成基于中位数的 m.inf
  m.inf <- my_m.inf.1a(x, which, dispersion)
  nms <- names(mt$tables[[which]])
  rownames(m.inf) <- nms
  
  # 按中位数从大到小排序
  ord <- order(m.inf[, 1], decreasing = TRUE)
  m.inf <- m.inf[ord, ]
  
  # 检查是否有所有中位数完全相同的情况
  if(length(unique(m.inf[,1])) == 1) {
    warning("All group medians are identical")
    groups <- rep(1, nrow(m.inf))
    names(groups) <- rownames(m.inf)
    return(list(av = x, groups = groups, nms = nms, ord = ord,
                m.inf = m.inf, sig.level = sig.level))
  }
  
  mMSE <- MSE/r
  dfr <- x$df.residual
  g <- nrow(m.inf)
  groups <- my_Partition(g, m.inf[, 1], mMSE, dfr, sig.level = sig.level, 
                         av = x, 1, rep(0, g), 0, rep(0, g))
  res <- list(av = x, groups = groups, nms = nms, ord = ord, 
              m.inf = m.inf, sig.level = sig.level)
  class(res) <- c("sk_esd", "list")
  invisible(res)
}

# 【修改 3】：入口函数支持非参数转换 (Rank Transformation) 与 version 参数激活
my_sk_esd <- function (x, alpha = 0.05, version = "np", ...) 
{
  x <- data.frame(x)
  melted_data <- reshape2::melt(x, id.vars = 0)
  
  # 如果开启 version="np"，先对数据进行秩变换 (Rank Conversion)
  if (version == "np") {
    melted_data$value <- rank(melted_data$value)
  }
  
  av <- aov(value ~ variable, data = melted_data)
  sk <- my_scottknott(av, which = "variable", dispersion = "mm", 
                      sig.level = alpha)
  names(sk$groups) <- rownames(sk$m.inf)
  class(sk) <- c(class(sk), "sk_esd")
  return(sk)
}