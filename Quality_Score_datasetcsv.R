#EXPLORATORY ANALYSIS OF INSTAGRAM DATA

#Import instagram data and save it in object called "di"
di <- read.csv("dataset.csv")

#Summarizing the data gile
head(di)

#Saving the unique users in the vecor "uni"
uni <- unique(di$USERNAME)

#Counting the number of unique users
length(uni)

#Save matrix of username and aggregate followers, following, likes, comments,
#number of tags, number of posts
group_user <- matrix(, nrow = NROW(uni), ncol = 7)
for (i in 1:NROW(uni)) {
    group_user[i,1] <- uni[i] 
    group_user[i,2] <- mean(di$FOLLOWERS[uni[i] == di$USERNAME],na.rm = TRUE)
    group_user[i,3] <- mean(di$FOLLOWING[uni[i] == di$USERNAME],na.rm = TRUE)
    group_user[i,4] <- mean(di$LIKES[uni[i] == di$USERNAME],na.rm = TRUE)
    group_user[i,5] <- mean(di$COMMENTS[uni[i] == di$USERNAME],na.rm = TRUE)
    group_user[i,6] <- sum(uni[i] == di$USERNAME)
    group_user[i,7] <- sd(di$LIKES[uni[i] == di$USERNAME],na.rm = TRUE)
}

colnames(group_user) = c("user", "followers", "following", "avg_likes", 
                         "coments", "posts", "std_likes")

g_user <- data.frame(group_user)

#Counting the number of NAs for standard deviation
#Or the number of users with just one photo and hence NA standard deviation
summary(group_user[,7])

#Create table with user name, likes, average and std dev of likes, 
#and quality score
score <- matrix(, nrow = NROW(di), ncol = 5)
for (i in 1:NROW(di)) { 
    score[i,1] <- di[i,1]
    score[i,2] <- di[i,4]
    score[i,3] <- g_user$avg_likes[uni[g_user$user] == di[i,1]]
    score[i,4] <- g_user$std_likes[uni[g_user$user] == di[i,1]]
    score[i,5] = (score[i,2]-score[i,3])/score[i,4]
}

colnames(score) = c("user", "likes", "avg_likes", "std_likes", "quality_score")

sc <- data.frame(score)

#Looking at summary stats of of quality scores
summary(sc$quality_score)

hist(sc$quality_score)

#Grouping quality scores by users and finding respective summary stats
q_score <- matrix(, nrow = NROW(uni), ncol = 12)
for (i in 1:NROW(di)) { 
    q_score[i,1] <- uni[i] 
    q_score[i,2] <- min(sc$quality_score[uni[i] == uni[sc$user]])
    q_score[i,3] <- quantile(sc$quality_score[uni[i] == uni[sc$user]],c(0.1),
                             na.rm = TRUE)
    q_score[i,4] <- quantile(sc$quality_score[uni[i] == uni[sc$user]],c(0.2),
                             na.rm = TRUE)
    q_score[i,5] <- quantile(sc$quality_score[uni[i] == uni[sc$user]],c(0.3),
                             na.rm = TRUE)
    q_score[i,6] <- quantile(sc$quality_score[uni[i] == uni[sc$user]],c(0.4),
                             na.rm = TRUE)
    q_score[i,7] <- quantile(sc$quality_score[uni[i] == uni[sc$user]],c(0.5),
                             na.rm = TRUE)
    q_score[i,8] <- quantile(sc$quality_score[uni[i] == uni[sc$user]],c(0.6),
                             na.rm = TRUE)
    q_score[i,9] <- quantile(sc$quality_score[uni[i] == uni[sc$user]],c(0.7),
                             na.rm = TRUE)
    q_score[i,10] <- quantile(sc$quality_score[uni[i] == uni[sc$user]],c(0.8),
                              na.rm = TRUE)
    q_score[i,11] <- quantile(sc$quality_score[uni[i] == uni[sc$user]],c(0.9),
                              na.rm = TRUE)
    q_score[i,12] <- max(sc$quality_score[uni[i] == uni[sc$user]])
}

colnames(q_score) = c("user", "min", "x10", "x20", "x30", "x40", "x50", "x60",
                      "x70", "x80", "x90", "max")

q_score = data.frame(q_score)
summary(q_score)
#Make unique score for each user (Min to 10th = 1, 10th to 20th = 2, 
#20th to 30th = 3, 30th to 40th = 4, 40th to 50th = 5, 50th to 60th = 6, 
#60th to 70th = 7, 70th to 80th = 8, 80th to 90th = 9, 90th to max = 10)
#For each quality score, find the number of tags, day, # of users in 
#photo, hour of the day and number of comments
fin_score <- matrix(, nrow = NROW(di), ncol = 7)
for (i in 1:NROW(di)) { 
    fin_score[i,1] <- di[i,1]
    fin_score[i,3] = di$NUM_OF_TAGS[i]
    fin_score[i,4] = di$DAY[i]
    fin_score[i,5] = di$USERS_IN_PHOTO[i]
    fin_score[i,6] = di$HOUR[i]
    fin_score[i,7] = di$COMMENTS[i]
    if (is.na(sc$quality_score[i]) == FALSE) {
        if (sc$quality_score[i] < q_score$x10[sc$user[i] == q_score$user]) { 
            fin_score[i,2] = 1
        } 
        else if (q_score$x10[sc$user[i] == q_score$user] <= 
                 sc$quality_score[i] & sc$quality_score[i] < 
                 q_score$x20[sc$user[i] == q_score$user]) {
            fin_score[i,2] = 2
        } 
        else if (q_score$x20[sc$user[i] == q_score$user] <= 
                 sc$quality_score[i] & sc$quality_score[i] < 
                 q_score$x30[sc$user[i] == q_score$user]) {
            fin_score[i,2] = 3
        } 
        else if (q_score$x30[sc$user[i] == q_score$user] <= 
                 sc$quality_score[i] & sc$quality_score[i] < 
                 q_score$x40[sc$user[i] == q_score$user]) {
            fin_score[i,2] = 4
        } 
        else if (q_score$x40[sc$user[i] == q_score$user] <= 
                 sc$quality_score[i] & sc$quality_score[i] < 
                 q_score$x50[sc$user[i] == q_score$user]) {
            fin_score[i,2] = 5
        } 
        else if (q_score$x50[sc$user[i] == q_score$user] <= 
                 sc$quality_score[i] & sc$quality_score[i] < 
                 q_score$x60[sc$user[i] == q_score$user]) {
            fin_score[i,2] = 6
        } 
        else if (q_score$x60[sc$user[i] == q_score$user] <= 
                 sc$quality_score[i] & sc$quality_score[i] < 
                 q_score$x70[sc$user[i] == q_score$user]) {
            fin_score[i,2] = 7
        } 
        else if (q_score$x70[sc$user[i] == q_score$user] <= 
                 sc$quality_score[i] & sc$quality_score[i] < 
                 q_score$x80[sc$user[i] == q_score$user]) {
            fin_score[i,2] = 8
        } 
        else if (q_score$x80[sc$user[i] == q_score$user] <= 
                 sc$quality_score[i] & sc$quality_score[i] < 
                 q_score$x90[sc$user[i] == q_score$user]) {
            fin_score[i,2] = 9
        } 
        else if (sc$quality_score[i] >= q_score$x90[sc$user[i] == 
                                                    q_score$user]) {
            fin_score[i,2] = 10
        }
    }
} 

colnames(fin_score) = c("user", "quality_score", "num_tags", "day", 
                        "users_in_photo", "hour", "num_comments")

fin_score = data.frame(fin_score)

#Looking at the distribution of scores. Counting the number of each respective
#quality score
qsc_dist <- matrix(, nrow = 2, ncol = 10)
for (i in 1:10) {
    qsc_dist[1,i] = i
    qsc_dist[2,i] = sum(fin_score$quality_score == i, na.rm = TRUE)
}

#Plotting distribution of quality scores
barplot(qsc_dist[2,])

#Average quality score by day
qsc_day <- matrix(, nrow = 1, ncol = 7)
for (i in 1:7) {
    qsc_day[1,i] = mean(fin_score$quality_score[fin_score$day== i], 
                        na.rm = TRUE)
}

colnames(qsc_day) = c("Sun", "Mon", "Tues", "Wed", "Thurs", "Fri", "Sat")

#Plotting quality score by day
barplot(qsc_day, main = "Quality score by day")

#Average quality score by hour
qsc_hr <- matrix(, nrow = 1, ncol = 24)
for (i in 1:24) {
    qsc_hr[1,i] = mean(fin_score$quality_score[fin_score$hour== i-1], 
                       na.rm = TRUE)
}

colnames(qsc_hr) = c("0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", 
                     "11", "12", "13", "14", "15", "16", "17", "18", "19", 
                     "20","21", "22", "23")

#Plotting quality score by hour
barplot(qsc_hr, main = "Quality score by hour")

#Average quality score by number of tags
qsc_tags <- matrix(, nrow = 1, ncol = 15)
for (i in 1:15) {
    qsc_tags[1,i] = mean(fin_score$quality_score[fin_score$num_tags == i], 
                         na.rm = TRUE)
}

colnames(qsc_tags) = c("1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", 
                       "12", "13", "14", "15")

#Plotting score by number of tags
barplot(qsc_tags, main = "Quality score by number of tags")

#Average quality score by number of users in photo
qsc_num_users <- matrix(, nrow = 1, ncol = 10)
for (i in 1:10) {
    qsc_num_users[1,i] = mean(fin_score$quality_score[fin_score$users_in_photo 
                                                      == i],na.rm = TRUE)
}

colnames(qsc_num_users) = c("1", "2", "3", "4", "5", "6", "7", "8", "9", "10")

#Plotting score by number of users in photo
barplot(qsc_num_users, main = "Quality score by number of users in photo")

#Comparing number of comments and quality score in photos
plot(fin_score$num_comments, fin_score$quality_score)

#Adding raw and scaled quality scores to the "di" dataframe
di$raw_qsc <- sc$quality_score
di$scale_qsc <- fin_score$quality_score

write.csv(di, "quality_score.csv")
