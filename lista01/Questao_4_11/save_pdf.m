function [] = save_pdf(h, file_name)
set(h,'Units','Inches');
pos = get(h,'Position');
set(h,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)])
print(file_name, '-dpdf', '-r0');
end