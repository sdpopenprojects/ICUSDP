library("ScottKnottESD")
library("reshape2")
library("car")
library("effsize")


my_m.inf.1a <- function (x, which, dispersion = c("mm", "s", "se")) 
{
  switch(match.arg(dispersion), mm = {
    m.inf <- aggregate(x$model[, 1], by = list(x$model[[which]]), 
                       function(x) c(mean = mean(x), min = min(x), max = max(x)))[, 
                                                                                  2]
  }, s = {
    m.inf <- aggregate(x$model[, 1], by = list(x$model[[which]]), 
                       function(x) c(mean = mean(x), `m - s` = mean(x) - 
                                       sd(x), `m + s` = mean(x) + sd(x)))[, 2]
  }, se = {
    m.inf <- aggregate(x$model[, 1], by = list(x$model[[which]]), 
                       function(x) c(mean = mean(x), `m - se` = mean(x) - 
                                       (sd(x)/sqrt(length(x))), `m + se` = mean(x) + 
                                       (sd(x)/sqrt(length(x)))))[, 2]
  })
}

my_Partition <- function(g, means, mMSE, dfr, sig.level, av, k, group, ngroup, 
                         markg, g1 = g, sqsum = rep(0, g1)) {
  
  # 处理means完全相同的情况
  # if(length(unique(means[k:g])) == 1) {
  #   ngroup <- ngroup + 1
  #   group[k:g] <- ngroup
  #   return(group)
  # }
  
  # 计算 sqsum
  for (k1 in k:(g - 1)) {
    t1 <- sum(means[k:k1])
    k2 <- g - k1
    t2 <- sum(means[(k1 + 1):g])
    sqsum[k1] <- t1^2/(k1 - k + 1) + t2^2/k2 - (t1 + t2)^2/(g - k + 1)
  }
  
  # 处理sqsum相同的情况
  max_sqsum <- max(sqsum)
  ord1 <- which(sqsum == max_sqsum)
  if(length(ord1) > 1) {
    # 如果有多个相同的最大sqsum，选择中间位置
    ord1 <- ord1[ceiling(length(ord1)/2)]
  } else {
    ord1 <- ord1[1]
  }
  
  # 修改diff函数，增加对相同均值的处理
  diff <- function(k, g, av, means) {
    if (k == g) return(TRUE)
    if (means[k] == means[g]) return(TRUE)  # 如果均值相同，直接返回TRUE
    
    a <- av$model[av$model[, 2] == names(means[k]), 1]
    b <- av$model[av$model[, 2] == names(means[g]), 1]
    
    if (length(a) == 0 || length(b) == 0 || var(a) == 0 || var(b) == 0) return(FALSE)
    
    a <- av$model[av$model[, 2] == names(means[k]), 1]
    b <- av$model[av$model[, 2] == names(means[g]), 1]
    magnitude <- as.character(cohen.d(a, b)$magnitude)
    return(magnitude == "negligible")
    
    # tryCatch({
    #   d_result <- effsize::cohen.d(a, b)
    #   magnitude <- as.character(d_result$magnitude)
    #   return(magnitude == "negligible")
    # }, error = function(e) {
    #   return(FALSE)
    # })
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
  m <- as.vector(mt$tables[[which]])
  nms <- names(mt$tables[[which]])
  
  # 处理相同均值的排序问题
  ord <- order(m, decreasing = TRUE)
  # 确保相同均值的组保持稳定顺序
  m.inf <- my_m.inf.1a(x, which, dispersion)
  rownames(m.inf) <- nms
  m.inf <- m.inf[order(m.inf[, 1], decreasing = TRUE), ]
  
  # 检查是否有完全相同的情况
  if(length(unique(m.inf[,1])) == 1) {
    warning("All group means are identical")
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

my_sk_esd <- function (x, alpha = 0.05, ...) 
{
  x <- data.frame(x)
  av <- aov(value ~ variable, data = reshape2::melt(x, id.vars = 0))
  sk <- my_scottknott(av, which = "variable", dispersion = "s", 
                   sig.level = alpha)
  names(sk$groups) <- rownames(sk$m.inf)
  class(sk) <- c(class(sk), "sk_esd")
  return(sk)
}
