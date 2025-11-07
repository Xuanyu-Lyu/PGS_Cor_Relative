
### rels in law, new ###

library(reshape2)
library(data.table)
# 

# what you need for this to work:
# 1. file with pairs of parents and offspring, called "p_o", with columns "child_id" and "parent_id"
# 2. file with sibs, called "sibs", with columns "i_lnr" and "j_lnr" with ID codes for each sib in the pair
# 3. file with mates, called "mates", with columns "mor" for the woman and "far" for the man


### create unique IDs for each pair ###
# in my files, the ID codes all start with a letter, and then there are 9 digits. 
# the pair ID is then these sets of 9 digits pasted together, always ordered such that the highest number comes first.
# The key thing here is just that a given pair of people will always get the same ID. regardless of who of them 
# is in the first or second column in the file, so that it becomes easy to filter out duplicate pairs.

p_o$pair_id   <- paste(pmax(substr(p_o$child_id,2,10), substr(p_o$parent_id,2,10)),
                       pmin(substr(p_o$child_id,2,10), substr(p_o$parent_id,2,10)), sep = "_")
sibs$pair_id  <- paste(pmax(substr(sibs$i_lnr,2,10), substr(sibs$j_lnr,2,10)),
                       pmin(substr(sibs$i_lnr,2,10), substr(sibs$j_lnr,2,10)), sep = "_")
mates$pair_id <- paste(pmax(substr(mates$mor,2,10), substr(mates$far,2,10)),
                       pmin(substr(mates$mor,2,10), substr(mates$far,2,10)), sep = "_")

### sib_spo ###
# create a new file with pairs of "sib-spouses", by merging "sibs" and "mates"
# there are four different ways to merge here: matching either sib "i" or sib "j" with 
# either the woman or the man in the mate pair. I do all these four merges and then rbind it all.

# "allow.cartesian = T" is important here, as it makes it so that all the pairs are found. Otherwise,
# each person would only be matched to maximally one other person. But it is possible to have several
# siblings and thereby several sib-spouses (also your spouse can have several siblings)

# "all.x = F" is also important. This makes it so that people who don't have any relatives
# of the relevant type are excluded. Otherwise, they would show up with NA in the 
# column for their relative

# also note that "merge" is here the merge function within the data.table package. 
# it is different to the base R version of "merge" in several ways, I think. Mainly, it is quicker.

sib_spo_jf <- merge(sibs, mates,
                    by.x = "i_lnr", by.y = "mor", all.x = F, allow.cartesian = T)
sib_spo_jf <- sib_spo_jf[, c("j_lnr", "far")]

sib_spo_jm <- merge(sibs, mates,
                    by.x = "i_lnr", by.y = "far", all.x = F, allow.cartesian = T)
sib_spo_jm <- sib_spo_jm[, c("j_lnr", "mor")]

sib_spo_if <- merge(sibs, mates,
                    by.x = "j_lnr", by.y = "mor", all.x = F, allow.cartesian = T)
sib_spo_if <- sib_spo_if[, c("i_lnr", "far")]

sib_spo_im <- merge(sibs, mates,
                    by.x = "j_lnr", by.y = "far", all.x = F, allow.cartesian = T)
sib_spo_im <- sib_spo_im[, c("i_lnr", "mor")]

names(sib_spo_jf) <-  
  names(sib_spo_jm) <-
  names(sib_spo_if) <-
  names(sib_spo_im) <- c("sib", "spo")

sib_spo <- rbind(sib_spo_jf, 
                 sib_spo_jm, 
                 sib_spo_if, 
                 sib_spo_im)

sib_spo$pair_id <- paste(pmax(substr(sib_spo$sib,2,10), substr(sib_spo$spo,2,10)),
                         pmin(substr(sib_spo$sib,2,10), substr(sib_spo$spo,2,10)), sep = "_")
sib_spo <- sib_spo[!duplicated(sib_spo$pair_id),]

rm(sib_spo_jf, 
   sib_spo_jm, 
   sib_spo_if, 
   sib_spo_im)
gc()

### sib_spo_sib ###

sib_spo_sib_i <- merge(sib_spo, sibs,
                       by.x = "spo", by.y = "i_lnr", all.x = F, allow.cartesian = T)
sib_spo_sib_i <- sib_spo_sib_i[, c("sib", "j_lnr")]

sib_spo_sib_j <- merge(sib_spo, sibs,
                       by.x = "spo", by.y = "j_lnr", all.x = F, allow.cartesian = T)
sib_spo_sib_j <- sib_spo_sib_j[, c("sib", "i_lnr")]

names(sib_spo_sib_i) <- names(sib_spo_sib_j) <- c("spo_sib", "sib")

sib_spo_sib <- rbind(sib_spo_sib_i,
                     sib_spo_sib_j)

sib_spo_sib$pair_id <- paste(pmax(substr(sib_spo_sib$spo_sib,2,10), substr(sib_spo_sib$sib,2,10)),
                             pmin(substr(sib_spo_sib$spo_sib,2,10), substr(sib_spo_sib$sib,2,10)), sep = "_")
sib_spo_sib <- sib_spo_sib[!duplicated(sib_spo_sib$pair_id),]

rm(sib_spo_sib_i, 
   sib_spo_sib_j)
gc()

### spo_sib_spo ###

spo_sib_spo_m <- merge(sib_spo, mates,
                       by.x = "sib", by.y = "mor", all.x = F, allow.cartesian = T)
spo_sib_spo_m <- spo_sib_spo_m[, c("spo", "far")]

spo_sib_spo_f <- merge(sib_spo, mates,
                       by.x = "sib", by.y = "far", all.x = F, allow.cartesian = T)
spo_sib_spo_f <- spo_sib_spo_f[, c("spo", "mor")]

names(spo_sib_spo_m) <- names(spo_sib_spo_f) <- c("sib_spo", "spo")

spo_sib_spo <- rbind(spo_sib_spo_m,
                     spo_sib_spo_f)

spo_sib_spo$pair_id <- paste(pmax(substr(spo_sib_spo$sib_spo,2,10), substr(spo_sib_spo$spo,2,10)),
                             pmin(substr(spo_sib_spo$sib_spo,2,10), substr(spo_sib_spo$spo,2,10)), sep = "_")
spo_sib_spo <- spo_sib_spo[!duplicated(spo_sib_spo$pair_id),]

rm(spo_sib_spo_m, 
   spo_sib_spo_f)
gc()


### spo_sib_chi ###

spo_sib_chi <- merge(sib_spo, p_o,
                     by.x = "sib", by.y = "parent_id", all.x = F, allow.cartesian = T)
spo_sib_chi <- spo_sib_chi[, c("spo", "child_id")]

names(spo_sib_chi) <- c("sib_spo", "chi")

spo_sib_chi$pair_id <- paste(pmax(substr(spo_sib_chi$sib_spo,2,10), substr(spo_sib_chi$chi,2,10)),
                             pmin(substr(spo_sib_chi$sib_spo,2,10), substr(spo_sib_chi$chi,2,10)), sep = "_")
spo_sib_chi <- spo_sib_chi[!duplicated(spo_sib_chi$pair_id),]


### sib_spo_sib_spo ###
sib_spo_sib_spo_m1 <- merge(sib_spo_sib, mates,
                            by.x = "sib", by.y = "mor", all.x = F, allow.cartesian = T)
sib_spo_sib_spo_m1 <- sib_spo_sib_spo_m1[, c("spo_sib", "far")]
names(sib_spo_sib_spo_m1) <- c("sib_spo_sib", "spo")

sib_spo_sib_spo_m2 <- merge(sib_spo_sib, mates,
                            by.x = "spo_sib", by.y = "mor", all.x = F, allow.cartesian = T)
sib_spo_sib_spo_m2 <- sib_spo_sib_spo_m2[, c("sib", "far")]
names(sib_spo_sib_spo_m2) <- c("sib_spo_sib", "spo")

sib_spo_sib_spo_f1 <- merge(sib_spo_sib, mates,
                            by.x = "sib", by.y = "far", all.x = F, allow.cartesian = T)
sib_spo_sib_spo_f1 <- sib_spo_sib_spo_f1[, c("spo_sib", "mor")]
names(sib_spo_sib_spo_f1) <- c("sib_spo_sib", "spo")

sib_spo_sib_spo_f2 <- merge(sib_spo_sib, mates,
                            by.x = "spo_sib", by.y = "far", all.x = F, allow.cartesian = T)
sib_spo_sib_spo_f2 <- sib_spo_sib_spo_f2[, c("sib", "mor")]
names(sib_spo_sib_spo_f2) <- c("sib_spo_sib", "spo")

sib_spo_sib_spo <- rbind(sib_spo_sib_spo_m1,
                         sib_spo_sib_spo_m2,
                         sib_spo_sib_spo_f1,
                         sib_spo_sib_spo_f2)

sib_spo_sib_spo$pair_id <- paste(pmax(substr(sib_spo_sib_spo$sib_spo_sib,2,10), substr(sib_spo_sib_spo$spo,2,10)),
                                 pmin(substr(sib_spo_sib_spo$sib_spo_sib,2,10), substr(sib_spo_sib_spo$spo,2,10)), sep = "_")
sib_spo_sib_spo <- sib_spo_sib_spo[!duplicated(sib_spo_sib_spo$pair_id),]

rm(sib_spo_sib_spo_m1, 
   sib_spo_sib_spo_m2, 
   sib_spo_sib_spo_f1, 
   sib_spo_sib_spo_f2)
gc()


### sib_spo_sib_chi ###
sib_spo_sib_chi_1 <- merge(sib_spo_sib, p_o,
                           by.x = "sib", by.y = "parent_id", all.x = F, allow.cartesian = T)
sib_spo_sib_chi_1 <- sib_spo_sib_chi_1[, c("spo_sib", "child_id")]
names(sib_spo_sib_chi_1) <- c("sib_spo_sib", "chi")

sib_spo_sib_chi_2 <- merge(sib_spo_sib, p_o,
                           by.x = "spo_sib", by.y = "parent_id", all.x = F, allow.cartesian = T)
sib_spo_sib_chi_2 <- sib_spo_sib_chi_2[, c("sib", "child_id")]
names(sib_spo_sib_chi_2) <- c("sib_spo_sib", "chi")

sib_spo_sib_chi <- rbind(sib_spo_sib_chi_1,
                         sib_spo_sib_chi_2)

sib_spo_sib_chi$pair_id <- paste(pmax(substr(sib_spo_sib_chi$sib_spo_sib,2,10), substr(sib_spo_sib_chi$chi,2,10)),
                                 pmin(substr(sib_spo_sib_chi$sib_spo_sib,2,10), substr(sib_spo_sib_chi$chi,2,10)), sep = "_")
sib_spo_sib_chi <- sib_spo_sib_chi[!duplicated(sib_spo_sib_chi$pair_id),]

rm(sib_spo_sib_chi_1, 
   sib_spo_sib_chi_2)
gc()


### sib_spo_sib_spo_sib ###
sib_spo_sib_spo_sib1 <- merge(sib_spo_sib_spo, sibs,
                              by.x = "spo", by.y = "i_lnr", all.x = F, allow.cartesian = T)
sib_spo_sib_spo_sib1 <- sib_spo_sib_spo_sib1[, c("sib_spo_sib", "j_lnr")]
names(sib_spo_sib_spo_sib1) <- c("spo_sib_spo_sib", "sib")

sib_spo_sib_spo_sib2 <- merge(sib_spo_sib_spo, sibs,
                              by.x = "spo", by.y = "j_lnr", all.x = F, allow.cartesian = T)
sib_spo_sib_spo_sib2 <- sib_spo_sib_spo_sib2[, c("sib_spo_sib", "i_lnr")]
names(sib_spo_sib_spo_sib2) <- c("spo_sib_spo_sib", "sib")

sib_spo_sib_spo_sib <- rbind(sib_spo_sib_spo_sib1,
                             sib_spo_sib_spo_sib2)

sib_spo_sib_spo_sib$pair_id <- paste(pmax(substr(sib_spo_sib_spo_sib$spo_sib_spo_sib,2,10), substr(sib_spo_sib_spo_sib$sib,2,10)),
                                     pmin(substr(sib_spo_sib_spo_sib$spo_sib_spo_sib,2,10), substr(sib_spo_sib_spo_sib$sib,2,10)), sep = "_")
sib_spo_sib_spo_sib <- sib_spo_sib_spo_sib[!duplicated(sib_spo_sib_spo_sib$pair_id),]

rm(sib_spo_sib_spo_sib1, 
   sib_spo_sib_spo_sib2)
gc()


### spo_sib_spo_sib_spo ###
spo_sib_spo_sib_spo1 <- merge(sib_spo_sib_spo, mates,
                              by.x = "sib_spo_sib", by.y = "mor", all.x = F, allow.cartesian = T)
spo_sib_spo_sib_spo1 <- spo_sib_spo_sib_spo1[, c("spo", "far")]
names(spo_sib_spo_sib_spo1) <- c("sib_spo_sib_spo", "spo")

spo_sib_spo_sib_spo2 <- merge(sib_spo_sib_spo, mates,
                              by.x = "sib_spo_sib", by.y = "far", all.x = F, allow.cartesian = T)
spo_sib_spo_sib_spo2 <- spo_sib_spo_sib_spo2[, c("spo", "mor")]
names(spo_sib_spo_sib_spo2) <- c("sib_spo_sib_spo", "spo")

spo_sib_spo_sib_spo <- rbind(spo_sib_spo_sib_spo1,
                             spo_sib_spo_sib_spo2)

spo_sib_spo_sib_spo$pair_id <- paste(pmax(substr(spo_sib_spo_sib_spo$sib_spo_sib_spo,2,10), substr(spo_sib_spo_sib_spo$spo,2,10)),
                                     pmin(substr(spo_sib_spo_sib_spo$sib_spo_sib_spo,2,10), substr(spo_sib_spo_sib_spo$spo,2,10)), sep = "_")
spo_sib_spo_sib_spo <- spo_sib_spo_sib_spo[!duplicated(spo_sib_spo_sib_spo$pair_id),]

rm(spo_sib_spo_sib_spo1, 
   spo_sib_spo_sib_spo2)
gc()

### spo_sib_spo_sib_chi ###
spo_sib_spo_sib_chi <- merge(sib_spo_sib_spo, p_o,
                             by.x = "sib_spo_sib", by.y = "parent_id", all.x = F, allow.cartesian = T)
spo_sib_spo_sib_chi <- spo_sib_spo_sib_chi[, c("spo", "child_id")]
names(spo_sib_spo_sib_chi) <- c("sib_spo_sib_spo", "chi")

spo_sib_spo_sib_chi$pair_id <- paste(pmax(substr(spo_sib_spo_sib_chi$sib_spo_sib_spo,2,10), substr(spo_sib_spo_sib_chi$chi,2,10)),
                                     pmin(substr(spo_sib_spo_sib_chi$sib_spo_sib_spo,2,10), substr(spo_sib_spo_sib_chi$chi,2,10)), sep = "_")
spo_sib_spo_sib_chi <- spo_sib_spo_sib_chi[!duplicated(spo_sib_spo_sib_chi$pair_id),]


### par_sib_spo_sib_chi ###
par_sib_spo_sib_chi <- merge(sib_spo_sib_chi, p_o,
                             by.x = "sib_spo_sib", by.y = "parent_id", all.x = F, allow.cartesian = T)
par_sib_spo_sib_chi <- par_sib_spo_sib_chi[, c("chi", "child_id")]
names(par_sib_spo_sib_chi) <- c("sib_spo_sib_chi", "chi")

par_sib_spo_sib_chi$pair_id <- paste(pmax(substr(par_sib_spo_sib_chi$sib_spo_sib_chi,2,10), substr(par_sib_spo_sib_chi$chi,2,10)),
                                     pmin(substr(par_sib_spo_sib_chi$sib_spo_sib_chi,2,10), substr(par_sib_spo_sib_chi$chi,2,10)), sep = "_")
par_sib_spo_sib_chi <- par_sib_spo_sib_chi[!duplicated(par_sib_spo_sib_chi$pair_id),]


### spo_sib_spo_sib_spo_sib ###
spo_sib_spo_sib_spo_sib1 <- merge(spo_sib_spo_sib_spo, sibs,
                                  by.x = "sib_spo_sib_spo", by.y = "i_lnr", all.x = F, allow.cartesian = T)
spo_sib_spo_sib_spo_sib1 <- spo_sib_spo_sib_spo_sib1[, c("spo", "j_lnr")]
names(spo_sib_spo_sib_spo_sib1) <- c("spo_sib_spo_sib_spo", "sib")

spo_sib_spo_sib_spo_sib2 <- merge(spo_sib_spo_sib_spo, sibs,
                                  by.x = "sib_spo_sib_spo", by.y = "j_lnr", all.x = F, allow.cartesian = T)
spo_sib_spo_sib_spo_sib2 <- spo_sib_spo_sib_spo_sib2[, c("spo", "i_lnr")]
names(spo_sib_spo_sib_spo_sib2) <- c("spo_sib_spo_sib_spo", "sib")

spo_sib_spo_sib_spo_sib <- rbind(spo_sib_spo_sib_spo_sib1,
                                 spo_sib_spo_sib_spo_sib2)

spo_sib_spo_sib_spo_sib$pair_id <- paste(pmax(substr(spo_sib_spo_sib_spo_sib$spo_sib_spo_sib_spo,2,10), substr(spo_sib_spo_sib_spo_sib$sib,2,10)),
                                         pmin(substr(spo_sib_spo_sib_spo_sib$spo_sib_spo_sib_spo,2,10), substr(spo_sib_spo_sib_spo_sib$sib,2,10)), sep = "_")
spo_sib_spo_sib_spo_sib <- spo_sib_spo_sib_spo_sib[!duplicated(spo_sib_spo_sib_spo_sib$pair_id),]

rm(spo_sib_spo_sib_spo_sib1,
   spo_sib_spo_sib_spo_sib2)
gc()


### sib_spo_sib_spo_sib_chi ###
sib_spo_sib_spo_sib_chi <- merge(sib_spo_sib_spo_sib, p_o,
                                 by.x = "spo_sib_spo_sib", by.y = "parent_id", all.x = F, allow.cartesian = T)
sib_spo_sib_spo_sib_chi <- sib_spo_sib_spo_sib_chi[, c("sib", "child_id")]
names(sib_spo_sib_spo_sib_chi) <- c("sib_spo_sib_spo_sib", "chi")

sib_spo_sib_spo_sib_chi$pair_id <- paste(pmax(substr(sib_spo_sib_spo_sib_chi$sib_spo_sib_spo,2,10), substr(sib_spo_sib_spo_sib_chi$chi,2,10)),
                                         pmin(substr(sib_spo_sib_spo_sib_chi$sib_spo_sib_spo,2,10), substr(sib_spo_sib_spo_sib_chi$chi,2,10)), sep = "_")
sib_spo_sib_spo_sib_chi <- sib_spo_sib_spo_sib_chi[!duplicated(sib_spo_sib_spo_sib_chi$pair_id),]



### connect everything ###

names(mates) <- 
  names(sib_spo) <- 
  names(sib_spo_sib) <- 
  names(spo_sib_spo) <- 
  names(spo_sib_chi) <- 
  names(sib_spo_sib_spo) <- 
  names(sib_spo_sib_chi) <- 
  names(sib_spo_sib_spo_sib) <- 
  names(spo_sib_spo_sib_spo) <- 
  names(spo_sib_spo_sib_chi) <- 
  names(par_sib_spo_sib_chi) <- 
  names(sib_spo_sib_spo_sib_chi) <- 
  names(spo_sib_spo_sib_spo_sib) <- c("i", "j", "pair_id")

mates$type                   <- "mates"
sib_spo$type                 <- "sib_spo"
sib_spo_sib$type             <- "sib_spo_sib"
spo_sib_spo$type             <- "spo_sib_spo"
spo_sib_chi$type             <- "spo_sib_chi"
sib_spo_sib_spo$type         <- "sib_spo_sib_spo"
sib_spo_sib_chi$type         <- "sib_spo_sib_chi"
sib_spo_sib_spo_sib$type     <- "sib_spo_sib_spo_sib"
spo_sib_spo_sib_spo$type     <- "spo_sib_spo_sib_spo"
spo_sib_spo_sib_chi$type     <- "spo_sib_spo_sib_chi"
par_sib_spo_sib_chi$type     <- "par_sib_spo_sib_chi"
sib_spo_sib_spo_sib_chi$type <- "sib_spo_sib_spo_sib_chi"
spo_sib_spo_sib_spo_sib$type <- "spo_sib_spo_sib_spo_sib"

ril <- rbind(mates, 
             sib_spo, 
             sib_spo_sib, 
             spo_sib_spo, 
             spo_sib_chi, 
             sib_spo_sib_spo, 
             sib_spo_sib_chi, 
             sib_spo_sib_spo_sib, 
             spo_sib_spo_sib_spo, 
             spo_sib_spo_sib_chi, 
             par_sib_spo_sib_chi, 
             sib_spo_sib_spo_sib_chi, 
             spo_sib_spo_sib_spo_sib)

ril <- ril[!duplicated(ril$pair_id),] 
# this takes out duplicate pairs, in such a way that if the same pair fits into several
# categories, they are only kept for the closest among these categories.

# here, it's a good idea to see if there are any of the identified pairs 
# who are also biologically related, and take them out.

# write.csv(ril, "inlaws.csv", row.names = F)




