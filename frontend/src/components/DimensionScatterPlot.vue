<template>
  <div class="chart-container">
    <div v-if="currentDimensions.length === 0" class="no-data-message">
      No columns selected. Please select columns to visualize.
    </div>
    <div ref="chart" class="chart"></div>
  </div>
</template>

<script>
import * as echarts from "echarts";

export default {
  name: "DimensionScatterPlot",
  props: {
    dimensions: {
      type: Array,
      required: true,
    },
    combinedDimension: {
      type: Array,
      default: () => [],
    },
  },
  data() {
    return {
      chart: null,
      lastBrushSelection: null,
    };
  },
  computed: {
    currentDimensions() {
      // Use combinedDimension if it's not empty, otherwise use dimensions
      return this.combinedDimension.length > 0
        ? this.combinedDimension
        : this.dimensions;
    },
  },
  mounted() {
    this.initChart();
    window.addEventListener("resize", this.resizeChart);
  },
  beforeDestroy() {
    window.removeEventListener("resize", this.resizeChart);
    if (this.chart) {
      this.chart.dispose();
      this.chart = null;
    }
  },
  methods: {
    initChart() {
      if (this.currentDimensions.length > 0) {
        this.chart = echarts.init(this.$refs.chart);
        this.updateChart();
      }
      window.addEventListener("resize", this.resizeChart);
    },
    updateChart() {
      if (!this.chart) return;

      if (this.currentDimensions.length === 0) {
        this.chart.clear();
        return;
      }

      const allClusters = [
        ...new Set(
          this.currentDimensions.flatMap((d) => d.data.map((item) => item.cluster))
        ),
      ];
      const colors = [
        "#ff4d4f",
        "#ffa940",
        "#ffec3d",
        "#73d13d",
        "#40a9ff",
        "#597ef7",
        "#9254de",
        "#f759ab",
      ];

      const series = [];

      this.currentDimensions.forEach((dimension) => {
        const dataByCluster = {};
        dimension.data.forEach((item) => {
          if (!dataByCluster[item.cluster]) {
            dataByCluster[item.cluster] = [];
          }
          dataByCluster[item.cluster].push({
            name: item.value,
            value: item.position,
            symbolSize: Math.sqrt(item.count) * 5,
            count: item.count,
            cluster: item.cluster,
            dimension: dimension.name,
          });
        });

        Object.keys(dataByCluster).forEach((clusterName) => {
          series.push({
            name: clusterName,
            type: "scatter",
            data: dataByCluster[clusterName],
            itemStyle: {
              color: this.getColorForCluster(clusterName, allClusters, colors),
              opacity: 0.8,
            },
            large: true,
            largeThreshold: 5000,
            progressive: 300,
            progressiveThreshold: 3000,
          });
        });
      });

      const option = {
        legend: {
          data: allClusters,
          top: "bottom",
          selectedMode: true, // Set to false if you want to disable toggling clusters
          icon: "circle",
          textStyle: {
            color: "#000",
            fontSize: 10,
          },
        },
        tooltip: {
          trigger: "item",
          formatter: (params) => {
            return `Dimension: ${params.data.dimension}<br/>Value: ${params.data.name}<br/>Count: ${params.data.count}<br/>Cluster: ${params.data.cluster}`;
          },
        },
        xAxis: {
          type: "value",
          scale: true,
          axisLine: { show: false },
          axisTick: { show: false },
          axisLabel: { show: false },
          splitLine: { show: false },
        },
        yAxis: {
          type: "value",
          scale: true,
          axisLine: { show: false },
          axisTick: { show: false },
          axisLabel: { show: false },
          splitLine: { show: false },
        },
        grid: {
          top: "3%",
          left: "3%",
          right: "3%",
          bottom: "3%",
          containLabel: false,
        },
        series: series,
        toolbox: {
          feature: {
            dataZoom: { yAxisIndex: "none" },
            restore: {},
            saveAsImage: {},
          },
        },
        dataZoom: [
          { type: "inside", xAxisIndex: [0], start: 0, end: 100 },
          { type: "inside", yAxisIndex: [0], start: 0, end: 100 },
        ],
        brush: {
          toolbox: ["rect", "polygon", "keep", "clear"],
          xAxisIndex: 0,
        },
      };

      this.chart.setOption(option, true);
      this.chart.off("brushSelected"); // Remove previous event listeners
      this.chart.off("brushEnd");
      this.chart.on("brushSelected", this.handleBrushSelected);
      this.chart.on("brushEnd", this.handleBrushEnd);
    },

    getColorForCluster(clusterName, allClusters, colors) {
      const index = allClusters.indexOf(clusterName);
      return colors[index % colors.length];
    },
    handleBrushEnd() {
      console.log("brushEnd");
    },
    handleBrushSelected(params) {
      const selectedData = params.batch[0].selected;
      if (selectedData && selectedData.length > 0) {
        const currentSelection = JSON.stringify(selectedData);
        if (currentSelection !== this.lastSelection) {
          this.lastSelection = currentSelection;

          console.log("Selected data details:");
          const detailedSelection = selectedData.flatMap(
            (series, seriesIndex) =>
              series.dataIndex.map((dataIndex) => {
                const point = this.chart.getOption().series[seriesIndex].data[
                  dataIndex
                ];
                return {
                  dimension: point.dimension,
                  value: point.name,
                  count: point.count,
                  cluster: point.cluster,
                  position: point.value,
                };
              })
          );

          console.log(detailedSelection);
          this.$emit("data-selected", detailedSelection);
        }
      } else {
        console.log("No data selected");
        this.$emit("data-selected", null);
      }
    },
    resizeChart() {
      if (this.chart) {
        this.chart.resize();
      }
    },
  },
  watch: {
    currentDimensions: {
      handler(newDimensions) {
        if (newDimensions.length === 0) {
          if (this.chart) {
            this.chart.clear();
          }
        } else {
          if (!this.chart) {
            this.initChart();
          } else {
            this.updateChart();
          }
        }
      },
      deep: true,
    },
  },
};
</script>

<style scoped>
.chart-container {
  width: 100%;
  height: 450px; /* Adjust this value as needed */
}

.chart {
  width: 100%;
  height: 100%;
}
.dimensionCharts {
  padding: 5px;
  border-top: 1px solid #e4e7ed;
  border-bottom: 1px solid #e4e7ed;
}
</style>
