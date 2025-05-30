<template>
  <div class="group-overview">
    <!-- <h3 class="overview-title">Group Overview</h3> -->
    <el-row :gutter="20" class="chart-row">
      <el-col :xs="24" :sm="12" class="chart-column">
        <div ref="totalValuesChart" class="chart"></div>
      </el-col>
      <el-col :xs="24" :sm="12" class="chart-column">
        <div ref="uniqueValuesChart" class="chart"></div>
      </el-col>
    </el-row>
    <table class="group-table" v-show="groupTableShown">
      <thead>
        <tr>
          <th>Group</th>
          <th>Size</th>
          <th>Unique Values</th>
        </tr>
      </thead>
      <tbody>
        <tr v-for="group in groups" :key="group.name">
          <td>{{ group.name }}</td>
          <td>{{ group.count }}</td>
          <td>{{ group.items.length }}</td>
        </tr>
      </tbody>
    </table>
  </div>
</template>

<script>
import * as echarts from 'echarts';

export default {
  name: 'GroupOverview',
  props: {
    groups: {
      type: Array,
      required: true
    },
    isVisible: {
      type: Boolean,
      default: false
    }
  },
  data() {
    return {
      totalValuesChart: null,
      uniqueValuesChart: null,
      groupTableShown: false,
    };
  },
  mounted() {
    window.addEventListener('resize', this.handleResize);
  },
  beforeDestroy() {
    window.removeEventListener('resize', this.handleResize);
    this.disposeCharts();
  },
  watch: {
    isVisible(newValue) {
      if (newValue) {
        this.$nextTick(() => {
          setTimeout(() => {
            this.initCharts();
          }, 100);
        });
      }
    },
    groups: {
      handler() {
        if (this.isVisible) {
          this.initCharts();
        }
      },
      deep: true
    }
  },
  methods: {
    initCharts() {
      this.disposeCharts();
      this.$nextTick(() => {
        this.totalValuesChart = echarts.init(this.$refs.totalValuesChart);
        this.uniqueValuesChart = echarts.init(this.$refs.uniqueValuesChart);

        const totalValuesOption = this.getChartOption('Total Values', this.groups.map(group => group.count), '#3a8ee6');
        const uniqueValuesOption = this.getChartOption('Unique Values', this.groups.map(group => group.items.length), '#67c23a');

        this.totalValuesChart.setOption(totalValuesOption);
        this.uniqueValuesChart.setOption(uniqueValuesOption);

        this.handleResize();
      });
    },
    handleResize() {
      if (this.totalValuesChart) {
        this.totalValuesChart.resize();
      }
      if (this.uniqueValuesChart) {
        this.uniqueValuesChart.resize();
      }
    },
    disposeCharts() {
      if (this.totalValuesChart) {
        this.totalValuesChart.dispose();
        this.totalValuesChart = null;
      }
      if (this.uniqueValuesChart) {
        this.uniqueValuesChart.dispose();
        this.uniqueValuesChart = null;
      }
    },
    getChartOption(title, data, color) {
      return {
        title: {
          text: title,
          left: 'center',
          textStyle: {
            fontSize: 14
          }
        },
        tooltip: {
          trigger: 'axis',
          axisPointer: {
            type: 'shadow'
          }
        },
        grid: {
          top: '10%',
          left: '3%',
          right: '4%',
          bottom: '10%',
          containLabel: true
        },
        xAxis: {
          type: 'category',
          data: this.groups.map(group => group.name),
          axisTick: {
            alignWithLabel: true
          },
          axisLabel: {
            fontSize: 12,
            rotate: 30,
            formatter: function (value) {
              return value.length > 10 ? value.slice(0, 10) + '...' : value;
            }
          }
        },
        yAxis: {
          type: 'value',
          axisLabel: {
            fontSize: 12
          }
        },
        series: [
          {
            name: title,
            type: 'bar',
            data: data,
            barWidth: '50%',
            itemStyle: {
              color: color
            }
          }
        ]
      };
    },
  },
};
</script>

<style scoped>
.group-overview {
  /* margin-bottom: 20px; */
}

.overview-title {
  font-size: 14px;
  margin-bottom: 10px;
  text-align: left;
}

.chart-container {
  display: flex;
  justify-content: space-between;
  margin-bottom: 20px;
}

.chart-row {
  display: flex;
  flex-wrap: nowrap;
}

.chart-column {
  flex: 1;
  min-width: 0;
  /* This allows the flex item to shrink below its content size */
}

.chart {
  height: 200px;
  width: 100%;
}

/* .chart {
  flex: 1;
  height: 300px;
  margin-right: 20px;
} */

.chart:last-child {
  margin-right: 0;
}

.group-table {
  width: 100%;
  border-collapse: collapse;
}

.group-table th,
.group-table td {
  border: 1px solid #ccc;
  padding: 8px;
  text-align: left;
}

.group-table th {
  background-color: #f9f9f9;
}

@media (max-width: 767px) {
  .chart-container {
    flex-direction: column;
  }

  .chart {
    width: 100%;
    margin-right: 0;
    margin-bottom: 20px;
  }

  .chart:last-child {
    margin-bottom: 0;
  }
}
</style>