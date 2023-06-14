


library(grid)
library(forestploter)

setwd('/Volumes/JasonWork/Projects/UKB_Proteomics/Results/Plots/Figure3/')
mydata = read.csv('Data/BaseProRS.csv')
#mydata = mydata[c(1:30),]

mydata$Disease = ifelse(mydata$Indent == 1, mydata$Disease, paste0("       ", mydata$Disease))
mydata = mydata[which(mydata$Select == 1), ]
bg_col_lst = mydata$Color_label1
bg_font_lst = mydata$font1
bg_fontszie_lst = rep(18,20)

mydata[,c(3)] = paste(rep(" ", 35), collapse = " ")

tm <- forest_theme(ci_pch = c(16, 16), # shape of middle point
                   ci_col = c('steelblue', 'tomato3'),
                   #ci_alpha = 1, # alpha of ci
                   #ci_lty = 1, # ci line type
                   ci_lwd = c(5, 5), # ci line thickness
                   #ci_Theight = 0.1, # end bar length
                   legend_value = c("PANEL", "ProRS+PANEL"),
                   refline_lwd = 5,
                   refline_lty = "solid",
                   refline_col = "darkolivegreen3",
                   # Vertical line width/type/color
                   vertline_lwd = 1,
                   vertline_lty = "dashed",
                   vertline_col = "grey40",
                   arrow_type = "closed",
                   core=list(fg_params=list(fontface=bg_font_lst,
                                            fontsize=bg_fontszie_lst),
                             bg_params=list(fill = bg_col_lst)))

p <- forest(mydata[, c(1, 2, 3)],
            est = list(mydata$base_diff, mydata$combo_diff_mean),
            lower = list(mydata$base_diff_lbd, mydata$combo_diff_lbd),
            upper = list(mydata$base_diff_ubd, mydata$combo_diff_ubd),
            ci_column = 3,
            sizes = 1.5,
            ref_line = 0,
            xlim = c(-0.29, 0.06),
            ticks_at = c(-0.25, -0.2, -0.15, -0.1, -0.05, 0, 0.05),
            vert_line = c(-0.2, -0.1),
            theme = tm
)

convertHeight(p$heights, "mm", valueOnly = TRUE) 
#p$heights <- rep(unit(9.5, "mm"), nrow(p)) #You can assign different value for each row if you want
p$heights <- rep(unit(10, "mm"), nrow(p)) #You can assign different value for each row if you want

plot(p)
#1250*1800
