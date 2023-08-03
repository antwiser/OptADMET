const renderBar = (dataPath = '/static/media/data/statistic.json', querySelectorID = 'propertyStacked', title) => {
    const colorConfig = new Map([
        ["Experimental Database", "#207BA1"],
        ["Expanded Database", "#E1982F"],
    ]);

    $.getJSON(dataPath, function(countData) {
        var chartDom = document.getElementById(querySelectorID);
        var myChart = echarts.init(chartDom);
        var option;
        const experimentalData = [];
        const expandedData = [];
        for (const [name, value] of Object.entries(countData['Experimental Database'])) {
            experimentalData.push(value);
        }
        for (const [name, value] of Object.entries(countData['Expanded Database'])) {
            expandedData.push(value);
        }
        option = {
            title: {
                text: title,
                left: "center",
            },
            tooltip: {
                trigger: 'axis',
                axisPointer: {
                    type: 'cross',
                }
            },
            grid: {
                top: "10%",
                containLab: true,
            },
            legend: {
                top: "4%",
                data: ['Experimental Database', 'Expanded Database']
            },
            yAxis: [{
                type: 'category',
                axisTick: {
                    alignWithLabel: true
                },
                data: ['LogD7.4', 'LogP', 'LogS', 'BBB', 'Caco-2', 'CYP1A2-inh', 'CYP2C19-inh', 'CYP2C9-inh', 'CYP2D6-inh', 'CYP2D6-sub', 'CYP3A4-inh', 'F30%', 'Pgp-inh', 'Pgp-sub', 'PPB', 'VD', 'T2/1', 'AMES', 'BCF', 'DILI', 'Eye Corrosion', 'Eye Irritation', 'FDAMDD', 'H-HT2', 'hERG', 'IGC50', 'NR-AhR', 'NR-AR', 'NR-AR-LBD', 'Respiratory', 'SR-ARE', 'SR-MMP']
            }],
            xAxis: [{
                type: 'value',
                // name: 'Expanded Database',
                min: 0,
                max: 15000,
                position: 'top',
                axisLabel: {
                    formatter: '{value}'
                }
            }],
            color: [colorConfig.get('Experimental Database'), colorConfig.get('Expanded Database')],
            series: [{
                name: 'Experimental Database',
                type: 'bar',
                data: experimentalData,
            }, {
                name: 'Expanded Database',
                type: 'bar',
                data: expandedData,
            }],
        };

        option && myChart.setOption(option);

        $(window).on('resize', function() {
            myChart.resize();
        });
    });
};