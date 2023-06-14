
library(scales)
library(ggplot2)
library(ggbeeswarm)
library(plyr)
setwd('/Volumes/JasonWork/Projects/UKB_Proteomics/Results/Plots/Figure5/Data/')
mydata = read.csv('F1_PLOT_Data1000.csv')
pro_code_lst = unique(mydata$Pro_code)
nb_pros = length(pro_code_lst)
id_lst = seq(1, nb_pros, 1)
mydata$id <- mapvalues(mydata$Pro_code, from=pro_code_lst, to=id_lst)
mydata$id = as.numeric(mydata$id)
angle <- 90 - 360 * (mydata$id-0.5)/nb_pros     # I substract 0.5 because the letter must have the angle of the center of the bars. Not extreme right(1) or extreme left (0)
mydata$hjust <- ifelse(angle < -90, 1, 0)
mydata$angle <- ifelse(angle < -90, angle+180, angle)
mydata = na.omit(mydata)


mydata$Pro_code = as.factor(mydata$Pro_code)
mydata$Pro_code = factor(mydata$Pro_code, levels = pro_code_lst)
mydata$shap_values = mydata$shap_values/abs(mydata$shap_values)*log(abs(mydata$shap_values) + 1)
#mydata$shap_values = (mydata$shap_values - mean(mydata$shap_values))/ sd(mydata$shap_values)
summary(mydata$shap_value)
mydata$shap_values[mydata$shap_values>0.6] = 0.6
mydata$shap_values[mydata$shap_values< -0.7] = -0.75
mydata$shap_values[mydata$Pro_code == 'AGR2'][mydata$shap_values[mydata$Pro_code == 'AGR2']>0.20] = 0.2
mydata$shap_values[mydata$Pro_code == 'SPINK4'][mydata$shap_values[mydata$Pro_code == 'SPINK4']>0.15] = 0.

mydata1 = mydata[mydata$TopPro == 1, ]
top_data = mydata1[mydata1$Pro_code %in% c('NEFL', 'LTBP2'), ]
medium_data = mydata1[!(mydata1$Pro_code %in% c('NEFL', 'LTBP2')), ]
bottom_data = mydata[mydata$TopPro == 0, ]

lbd = min(mydata$Pro_values, na.rm = TRUE)
ubd = max(mydata$Pro_values, na.rm = TRUE)

p = ggplot(mydata, aes(x = Pro_code, y = shap_values, color=Pro_values)) + 
  geom_hline(yintercept=-0.05, linetype="solid", color = "gray50", linewidth=1)+
  geom_hline(yintercept=-1.25, linetype="dashed", color = "gray50", linewidth=2)+
  geom_hline(yintercept=0, linetype="solid", color = "gray50", linewidth=1)+
  geom_hline(yintercept=0.05, linetype="solid", color = "gray50", linewidth=1)+
  geom_quasirandom(method='pseudorandom', dodge.width=0.5, alpha=.7, size = 0.85)+ 
  scale_colour_gradientn(colours = c("darkred", "red", "gray95", "steelblue", "blue"),
                         values = rescale(c(lbd, -0.5, 0., 0.5, ubd)))+
  ylim(-0.75, 0.65)+
  coord_polar()+
  theme_bw()+
  theme(legend.position = "none", 
        axis.title = element_blank(),
        panel.grid = element_blank(), 
        axis.text.x = element_blank(),
        plot.margin = unit(rep(0, 4), "cm"))+
  geom_text(data=top_data, aes(x=id, y = 0.62, label=Pro_code, hjust=hjust), color="black", size=7.5, fontface = 'bold', angle= top_data$angle, check_overlap = T)+
  geom_text(data=medium_data, aes(x=id, y = 0.45, label=Pro_code, hjust=hjust), color="black", size=7.5, fontface = 'bold', angle= medium_data$angle, check_overlap = T)+
  geom_text(data=bottom_data, aes(x=id, y = 0.25, label=Pro_code, hjust=hjust), color="black", size=3.5, angle= bottom_data$angle, check_overlap = T)

ggsave('ACD.png', width = 17.6, height = 17, device='png', dpi=150)


