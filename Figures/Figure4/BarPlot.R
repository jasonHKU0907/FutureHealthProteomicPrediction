
library("scales")
library("ggplot2")
setwd("/Volumes/JasonWork/Projects/UKB_Proteomics/Results/Plots/Figure4/Data")


mydata = read.csv("FeaImpRk4R.csv")
dis_code_lst = unique(mydata$DiseaseCode)
mydata$DiseaseCode = factor(mydata$DiseaseCode, levels = dis_code_lst)
order_df = mydata[1:34,]
pro_code_lst1 = order_df$Pro_code
pro_code_lst2 = order_df[order(order_df$ImpSum, decreasing = TRUE), 'Pro_code']
mydata$Pro_code = factor(mydata$Pro_code, levels = pro_code_lst2)

color_lst = c('gray80', 'palegreen3', 'skyblue1', 'brown2', 'salmon1', 'gray40', 'dodgerblue', 'green3', 'thistle2', 
              'orange', 'darkorange2', 'navajowhite1', 'lightslateblue', 'yellow3', 'dodgerblue4')

ggplot(mydata, aes(fill=DiseaseCode, y=ShapValue, x=Pro_code)) + 
  geom_bar(position="stack", stat="identity")+
  scale_fill_manual(values=color_lst)+
  theme_bw()+
  ylab("Stacked SHAP values")+
  guides(fill = guide_legend('Disease category', byrow = TRUE)) +
  theme(axis.title.x = element_blank(), 
        axis.text.x = element_text(angle = 45, hjust=1, vjust = 1.35, size = 20, color = 'gray15'),
        axis.ticks.x = element_blank(), 
        axis.title.y = element_text(size = 24, color = 'gray15'),
        axis.text.y = element_text(size = 18, color = 'gray15'),
        panel.grid.major = element_blank(), 
        panel.grid.minor = element_blank(),
        panel.border = element_blank(),
        legend.background = element_rect(fill='gray95'),
        legend.position=c(0.9, 0.62),
        legend.key.height= unit(.85, 'cm'),
        legend.key.width= unit(1, 'cm'),
        legend.title = element_text(color = 'gray15', size=30, vjust = 1),
        legend.text = element_text(color = 'gray15', size=18),
        legend.spacing.y = unit(0.15, "cm"),
        legend.spacing.x = unit(0.5, "cm"))+
  geom_text(data = mydata, aes(Pro_code, ImpSum + 0.03, label = mydata$NbTop15), size = 10, color = 'gray25',check_overlap = T)



