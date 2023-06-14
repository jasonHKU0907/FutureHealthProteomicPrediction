
library("scales")
library("ggplot2")
setwd("/Volumes/JasonWork/Projects/UKB_Proteomics/Results/Plots/Figure4/Data")

pro_imp_df = read.csv("FeaImpRanking.csv")
pro_imp_df = pro_imp_df[1:34, ]
pro_imp_df = pro_imp_df[order(pro_imp_df$ImpSum, decreasing = TRUE), ]
pro_f_lst = pro_imp_df$Pro_code[1:34]

dis_df = read.csv("Target_code.csv")
dis_df = dis_df[which(dis_df$Select == 1),]
dis_f_lst = dis_df$Disease


plotdf = read.csv("HeatmapMatrix.csv")
plotdf$Diseases <- factor(plotdf$Diseases, levels=dis_f_lst)
plotdf$Pro_code <- factor(plotdf$Pro_code, levels=rev(pro_f_lst))

bold.dis <- c("Diseases of Infections", "Cancer", "Blood and immune disorders", "Endocrine disorders", 
              "Mental and behavioural disorders", "Nervous system disorders", "Eye disorders", "Ear disorders",
              "Circulatory system disorders", "Respiratory system disorders", "Digestive system disorders", 
              "Skin disorders", "Musculoskeletal system disorders", "Genitourinary system disorders", "All-cause mortality")

bold.labels <- ifelse(levels(plotdf$Diseases) %in% bold.dis, yes = "bold", no = "plain")


ggplot(data =  plotdf, aes(x = Diseases, y = Pro_code)) + 
  geom_tile(aes(fill = HR), color = 'white', size = 0.25) +
  scale_fill_gradientn(colours=c("darkblue", "white", "salmon", "darkred"),
                       values=rescale(c(0, 1, 5, 33)),
                       guide="colorbar")+
  #geom_text(aes(label=InTop), size = 10, vjust = 0.8)+
  geom_text(aes(label=SigPval), size = 10, vjust = 0.8)+
  scale_y_discrete(position = "left")+
  #scale_y_discrete(sec.axis = sec_axis(~., name = "Y-axis Label on right side"))+
  theme(axis.text.x = element_text(angle = 45, hjust=1, size = 24, face = bold.labels),
        #axis.text.x = element_blank(),
        axis.text.y = element_text(size = 24),
        axis.title.x = element_blank(),
        axis.title.y = element_blank(),
        plot.margin = margin(0.25, 0.25, 0.25, 1, "cm"),
        legend.position = "none")


