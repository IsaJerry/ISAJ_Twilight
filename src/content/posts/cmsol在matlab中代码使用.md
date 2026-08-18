---
title: CMSOL在Matlab中代码使用
published: 2026-08-18T16:47:00.000+08:00
updated: 2026-08-18T17:01:00.000+08:00
description: 代码使用注意
tags:
  - COMSOL
draft: false
---
不要在matlab中直接创建plot等表格，应在COMSOL中创建然后在Matlab中调用。

如下所示：

```matlab
model.result.export('plot1').set('plotgroup', 'pg3');
    model.result.export('plot1').set('filename', filename);
    model.result.export('plot1').run;
```



```
images = xlsread("..\PycharmProjects\set_data\integrated_images\0.xlsx", 'images');
images = reshape(images, [100, 50, 50]);
soil_parameters = xlsread('..\PycharmProjects\set_data\integrated_images\0.xlsx', 'soil_parameters');
model = mphload("topology.mph");
for i = 1:100
    tic
    i;
    graphic = reshape(images(i, :,:), [50, 50]);
    Es = soil_parameters(i, 1)*1e6;
    Ps = soil_parameters(i, 2);
    rhos = soil_parameters(i, 3)*1e3;
    cell = [];
    flag = 1;
    for row = 1:50
        for col = 1:50
            if graphic(row, col) == 1
                cell(flag) = 50*(row-1)+col;
            end
        end
    end
    model.param.set('Es', num2str(Es));
    model.param.set('Ps', num2str(Ps));
    model.param.set('rhos', num2str(rhos));

    model.component('comp1').physics('solid').feature('lemm2').selection.set(cell);
    model.study('std1').run;

    %model.result.create('pg2', 'PlotGroup1D');
    %model.result('pg2').run;
    %model.result('pg2').create('glob1', 'Global');
    %model.result('pg2').feature('glob1').set('markerpos', 'datapoints');
    %model.result('pg2').feature('glob1').set('linewidth', 'preference');
    %model.result('pg2').feature('glob1').set('data', 'dset2');
    %model.result('pg2').feature('glob1').set('expr', {'solid.freq'});
    %model.result('pg2').feature('glob1').set('descr', {[native2unicode(hex2dec({'98' '91'}), 'unicode')  native2unicode(hex2dec({'73' '87'}), 'unicode') ]});
    %model.result('pg2').feature('glob1').set('unit', {'Hz'});
    %model.result('pg2').feature('glob1').set('xdatasolnumtype', 'outer');
    %model.result('pg2').run;
    %model.result.export.create('plot1', 'Plot');
    %model.result.export('plot1').set('plotgroup', 'pg2');
    
    model.result.export('plot1').set('filename', ['dispersion_curves/t',num2str(i),'.csv']);
    model.result.export('plot1').run;
    model.hist.disable;
    toc
end
```
