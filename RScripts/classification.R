A <- matrix(data = c(10, 14, 23, 19, 20, 11, 15, 24, 17, 21, 9, 12, 20, 16, 19, 8, 13, 17, 17, 20, 12, 15, 19, 15, 22), 5, 5, TRUE)
shifted_A <- A - 17
t_i <- vector()
t_j <- vector()
for (x in 1:5) {
  t_i <- append(t_i, sum(shifted_A[x,]))
  t_j <- append(t_j, sum(shifted_A[, x]))
}

n_i <- length(shifted_A[1,])
n_j <- length(shifted_A[, 1])

Ti2_ni <- round((t_i * t_i) / n_i, 2)
Ti2_nj <- round((t_j * t_j) / n_j, 2)

sum_ti2_ni <- sum(Ti2_ni)
sum_ti2_nj <- sum(Ti2_nj)

rows2 <- vector()
columns2 <- vector()

for (x_ in 1:5) {
  r <- shifted_A[x_,]
  c <- shifted_A[, x_]

  rows2 <- append(rows2, sum(r * r))
  columns2 <- append(columns2, sum(c * c))
}

sum_rows2 <- sum(rows2)
sum_columns2 <- sum(columns2)

convection_factor <- (sum(t_i)^2) / n_i
s <- sum_columns2 - convection_factor
s1 <- sum_ti2_ni - convection_factor
s2 <- sum_ti2_nj - convection_factor
s3 <- s - s1 - s2

dof_rows <- n_i - 1
dof_columns <- n_j - 1
dof_residual <- dof_columns * dof_rows

ms1 <- s1 / dof_rows
ms2 <- s2 / dof_columns
ms3 <- s3 / dof_residual

f_rows <- 0.0
f_columns <- 0.0

if (ms3 > ms1) {
  f_rows <- round(ms3 / ms1, 2)
} else {
  f_rows <- round(ms1 / ms3, 2)
}

if (ms3 > ms2) {
  f_columns <- round(ms3 / ms2, 2)
} else {
  f_columns <- round(ms2 / ms3, 2)
}

table1 <- data.frame()