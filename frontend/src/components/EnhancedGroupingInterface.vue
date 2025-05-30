<template>
  <el-card class="box-card">
    <el-steps
      :active="activeStep"
      finish-status="success"
      simple
      size="mini"
      style="font-size: 13px; "
    >
      <el-step size="mini" title="Data Overview" icon="el-icon-view" style="margin: 0; padding: 0;"></el-step>
      <el-step
        size="mini"
        title="Group Standardization"
        icon="el-icon-collection"
        style="margin: 0; padding: 0;"
      ></el-step>
    </el-steps>
    <div v-if="activeStep === 0" class="step-content">
      <div class="column-overview">
        <el-table
          :data="columnStatistics"
          style="width: 100%"
          height="200"
          size="mini"
          :fit="true"
          border
          stripe
        >
          <el-table-column
            prop="name"
            label="Column Name"
            min-width="180"
          ></el-table-column>
          <el-table-column
            prop="dataType"
            label="Data Type"
            min-width="120"
          ></el-table-column>
          <el-table-column
            prop="inconsistency"
            label="Inconsistency"
            min-width="120"
            sortable
          >
            <template slot-scope="scope">
              <el-progress
                size="mini"
                style="font-size: 12px !important"
                :percentage="parseFloat(scope.row.inconsistency)"
                :color="getColorForScore(parseFloat(scope.row.inconsistency))"
              >
              </el-progress>
            </template>
          </el-table-column>
          <el-table-column
            prop="uniqueValues"
            label="Unique Values"
            min-width="120"
            sortable
          ></el-table-column>
          <el-table-column
            label="Actions"
            min-width="120"
            align="center"
          >
            <template slot-scope="scope">
              <el-button
                @click="addColumn(scope.row.name)"
                size="mini"
                icon="el-icon-view"
                >View</el-button
              >
            </template>
          </el-table-column>
        </el-table>
      </div>
      <div class="column-selector">
        <div class="selected-columns">
          <el-tag
            v-for="column in selectedColumns"
            :key="column"
            closable
            @close="removeColumn(column)"
          >
            {{ column }}
          </el-tag>
        </div>
      </div>
      <div class="scatter-plot-container">
        <dimension-scatter-plot
          :dimensions="selectedDimensions"
          :combinedDimension="combinedDimension"
        ></dimension-scatter-plot>
      </div>
      <el-button
        size="mini"
        type="primary"
        @click="proceedToGrouping"
        :disabled="selectedColumns.length === 0"
        style="float: right; margin-top: 20px; margin-bottom: 10px"
      >
        Proceed to Standardization
      </el-button>
    </div>
    <div v-if="activeStep === 1" class="step-content">
      <el-row style="text-align: right">
        <div>
          <el-select v-model="sortMethod" placeholder="Sort by" size="mini">
            <el-option label="Size" value="size"></el-option>
            <el-option label="Inconsistency" value="inconsistency"></el-option>
          </el-select>
          <el-popover
            placement="bottom"
            width="400"
            trigger="hover"
            v-model="showInfoPopup"
          >
            <div class="info-popup">
              <el-alert
                v-if="contextualInsights.length > 0"
                title="Insights"
                type="warning"
                :closable="false"
                class="insight-panel"
              >
                <ul class="insight-list">
                  <li
                    v-for="(insight, index) in contextualInsights"
                    :key="index"
                  >
                    {{ insight }}
                  </li>
                </ul>
              </el-alert>

              <el-alert
                v-if="standardizationPreview"
                title="Standardization Preview"
                type="info"
                :closable="false"
                class="preview-panel"
              >
                <p>{{ standardizationPreview.preview }}</p>
              </el-alert>
            </div>
            <i
              slot="reference"
              class="el-icon-s-opportunity"
              style="margin-left: 5px; color: gray; cursor: pointer"
              @click="showInfoPopup = !showInfoPopup"
            ></i>
          </el-popover>
        </div>
      </el-row>

      <div class="grouped-dimensions-container">
        <grouped-dimension
          v-for="(dimension, index) in sortedDimensions"
          :key="index"
          :dimension="dimension"
          @standardize="handleStandardize"
          @view-in-table="
            (dimensionName, labelsToFilter) =>
              $emit('view-in-table', dimensionName, labelsToFilter)
          "
        />
      </div>
      <el-button
        type="primary"
        @click="goBackToOverview"
        size="mini"
        style="float: right; margin-bottom: 10px"
      >
        Back to Data Overview
      </el-button>
    </div>
  </el-card>
</template>

<script>
import GroupedDimension from "./GroupedDimension.vue";
import DimensionScatterPlot from "./DimensionScatterPlot.vue";

export default {
  emits: ["view-in-table", "update-grouped-dimensions", "standardize"],
  name: "EnhancedGroupingInterface",
  components: {
    GroupedDimension,
    DimensionScatterPlot,
  },
  props: {
    columnStatistics: {
      type: Array,
      required: true
    },
    dimensions: {
      type: Array,
      required: true,
    },
    qaTableData: {
      type: Array,
      required: true,
    },
  },
  data() {
    return {
      activeStep: 0,
      sortMethod: "size",
      standardizationPreview: null,
      showInfoPopup: false,
      selectedColumn: "",
      selectedColumns: [],
      combinedDimension: [],
    };
  },
  computed: {
    availableColumns() {
      return this.dimensions
        .map((d) => d.name)
        .filter((name) => !this.selectedColumns.includes(name));
    },
    selectedDimensions() {
      return this.dimensions.filter((d) =>
        this.selectedColumns.includes(d.name)
      );
    },
    sortedDimensions() {
      return [...this.dimensions]
        .sort((a, b) => {
          if (this.sortMethod === "size") {
            return b.data.length - a.data.length; // Sort in descending order
          }
          if (this.sortMethod === "inconsistency") {
            return (
              this.calculateInconsistency(b) - this.calculateInconsistency(a)
            ); // Sort in descending order
          }
          return 0;
        })
        .map((dim) => ({
          ...dim,
          groups: this.groupDataByClusters(dim.data).sort(
            (a, b) => b.count - a.count
          ), // Sort groups in descending order
        }));
    },
    contextualInsights() {
      const insights = [];
      this.sortedDimensions.forEach((dim) => {
        const largeGroups = dim.groups.filter((group) => group.count > 2);
        if (largeGroups.length > 0) {
          insights.push(
            `${dim.name} has ${largeGroups.length} groups with more than 2 different representations. Consider standardizing.`
          );
        }
      });
      return insights;
    },
  },
  methods: {
    async prepareGroupedData() {
      try {
        const response = await fetch('http://localhost:5010/api/prepare_grouped_data', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ 
            qa_table_data: this.qaTableData,
            selected_columns: this.selectedColumns 
          }),
        });

        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        console.log('===Grouped data:=== (in enhanced grouping interface)', data);
        this.combinedDimension = data.combinedDimension;
        this.$emit('update-grouped-dimensions', data.groupedDimensions);
      } catch (error) {
        console.error('Error preparing grouped data:', error);
      }
    },
    getColorForScore(score) {
      if (score < 0.3) return "#67C23A";
      if (score < 0.7) return "#E6A23C";
      return "#F56C6C";
    },
    addColumn(column) {
      if (!this.selectedColumns.includes(column)) {
        this.selectedColumns.push(column);
        this.prepareGroupedData();
      }
    },
    removeColumn(column) {
      this.selectedColumns = this.selectedColumns.filter((c) => c !== column);
      //   update selected dimensions
      this.prepareGroupedData();
    },
    handleStandardize(dimension, group) {
      console.log(`Standardize ${group} in ${dimension}`);
      this.standardizationPreview = {
        dimension,
        group,
        preview:
          "All values will be converted to μg/g. This will reduce unique values from 3 to 1.",
      };
      this.$emit("standardize", dimension, group);
    },
    calculateInconsistency(dimension) {
      const uniqueValues = new Set(dimension.data.map((item) => item.value));
      return uniqueValues.size / dimension.data.length;
    },
    groupDataByClusters(data) {
      const groups = {};
      data.forEach((item) => {
        if (!groups[item.cluster]) {
          groups[item.cluster] = { name: item.cluster, count: 0, items: [] };
        }
        groups[item.cluster].count += item.count;
        groups[item.cluster].items.push(item);
      });
      return Object.values(groups);
    },
    handleViewInTable(dimension, group) {
      console.log(`View ${group} from ${dimension} in table`);
      // Implement logic to highlight/filter main data table
    },
    proceedToGrouping() {
      this.activeStep = 1;
    },
    goBackToOverview() {
      this.activeStep = 0;
    },
  },
  watch: {
    selectedDimensions: {
      handler() {
        // This watcher ensures that the scatter plot updates when selectedDimensions changes
        this.$nextTick(() => {
          if (this.$refs.scatterPlot) {
            this.$refs.scatterPlot.updateChart();
          }
        });
      },
      deep: true,
    },
  },
};
</script>

<style scoped>
.box-card {
  width: 100%;
  height: 100%;
  /* Make the card take full height of its container */
  display: flex;
  flex-direction: column;
  overflow-y: auto;
  /* Enable vertical scrolling */
}

.scatter-plot-container {
  height: 400px;
  /* Adjust this value as needed */
  margin-bottom: 20px;
}

.card-content {
  flex: 1;
  display: flex;
  flex-direction: column;
  overflow: hidden;
  /* Hide overflow */
}

.el-card__body {
  flex: 1;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

.info-popup {
  max-height: 300px;
  overflow-y: auto;
}

.info-panels {
  max-height: 150px;
  /* Adjust this value as needed */
  overflow-y: auto;
  margin-bottom: 20px;
}

.insight-panel,
.preview-panel {
  margin-bottom: 10px;
}

.insight-list {
  padding-left: 20px;
  margin: 0;
  text-align: left;
}

.grouped-dimensions-container {
  flex: 1;
  overflow-y: auto;
  padding: 0 15px 15px;
}

/* Additional styles for GroupedDimension component */
::v-deep .el-collapse-item__header {
  font-size: 14px;
}

::v-deep .el-card__header {
  padding: 10px 20px;
}

::v-deep .el-card__body {
  padding: 0 0;
  display: flex;
  flex-direction: column;
  height: calc(100% - 50px);
  /* Adjust based on your header height */
}

::v-deep .el-tag {
  margin-right: 8px;
  margin-bottom: 8px;
}

.column-selector {
  display: flex;
  align-items: center;
  margin-top: 5px;
  margin-bottom: 10px;
  margin-left: 5px;
}

.selected-columns {
  display: inline-flex;
  flex-wrap: wrap;
  margin-left: 10px;
}

.selected-columns .el-tag {
  margin-right: 5px;
  margin-bottom: 5px;
}

.el-progress__text {
  font-size: 12px !important;
}

.grouped-dimensions-container {
  flex: 1;
  overflow-y: auto;
  padding: 0 15px 15px;
}
</style>
