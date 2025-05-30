<template>
  <el-collapse v-model="activeNames">
    <el-collapse-item :name="dimension.name">
      <template slot="title">
        <span>{{ dimension.name }}</span>
        <!-- <span style="float: right; padding: 0 8px;">
                    Total: {{ totalSize }}
                </span> -->
      </template>

      <group-overview
        :groups="sortedLocalGroups"
        :is-visible="isExpanded"
        @group-click="handleGroupClick"
      />

      <!-- Compact color legend and closed tags -->
      <div class="compact-legend">
        <span class="legend-label">Frequency:</span>
        <el-tag size="mini" class="high-frequency">High</el-tag>
        <el-tag size="mini" class="medium-frequency">Medium</el-tag>
        <el-tag size="mini" class="low-frequency">Low</el-tag>
      </div>
      <div class="closed-tags">
        <el-tag
          v-for="tag in closedTags"
          :key="tag.value"
          size="mini"
          @click.native="restoreTag(tag)"
        >
          {{ tag.value }}
        </el-tag>
      </div>

      <!-- New group creation -->
      <div class="new-group">
        <el-input
          v-model="newGroupName"
          size="mini"
          placeholder="New group name"
        >
          <el-button
            slot="append"
            icon="el-icon-plus"
            @click="createNewGroup"
          ></el-button>
        </el-input>
      </div>

      <div class="groupCardLists">
        <draggable
          v-model="sortedLocalGroups"
          group="groups"
          @start="drag = true"
          @end="drag = false"
        >
          <div
            v-for="(group, groupIndex) in localGroups"
            :key="groupIndex"
            class="group-card"
          >
            <div class="group-header">
              <el-input
                size="mini"
                v-model="group.name"
                @change="handleNameChange(groupIndex, $event)"
                class="editable-category-name"
              >
                <el-button
                  slot="append"
                  icon="el-icon-edit"
                  size="mini"
                ></el-button>
              </el-input>
              <div class="group-header-controls">
                <span class="group-size">Size: {{ group.count }}</span>
                <el-button
                  type="text"
                  icon="el-icon-edit"
                  size="mini"
                  @click="handleStandardize(group)"
                ></el-button>
                <el-button
                  type="text"
                  icon="el-icon-view"
                  size="mini"
                  @click="handleViewInTable(group)"
                ></el-button>
              </div>
            </div>
            <div class="group-content">
              <draggable
                v-model="group.items"
                :group="{ name: 'tags', pull: 'clone', put: true }"
                @change="handleTagMove"
                :sort="false"
                class="group-labels"
              >
                <el-tag
                  v-for="item in group.items"
                  :key="item.value"
                  :class="getTagClass(item.count, group.count)"
                  size="mini"
                  class="tag"
                  closable
                  @close="closeTag(group, item)"
                >
                  <span class="tag-value">{{ item.value }}</span>
                  <span class="tag-count">{{ item.count }}</span>
                </el-tag>
              </draggable>
            </div>
          </div>
        </draggable>
      </div>
    </el-collapse-item>
  </el-collapse>
</template>

<script>
import draggable from "vuedraggable";
import GroupOverview from "./GroupOverview.vue";

export default {
  name: "GroupedDimension",
  components: {
    GroupOverview,
    // CropNutrientMatrix,
    draggable,
  },
  props: {
    dimension: {
      type: Object,
      required: true,
    },
    isFirstCategory: {
      type: Boolean,
      default: false,
    },
  },
  data() {
    return {
      activeNames: [],
      closedTags: [],
      newGroupName: "",
      localGroups: this.groupDataByClusters(this.dimension.data),
      drag: false,
    };
  },
  computed: {
    totalSize() {
      return this.dimension.data.reduce((acc, item) => acc + item.count, 0);
    },
    isExpanded() {
      return this.activeNames.includes(this.dimension.name);
    },
    sortedLocalGroups() {
      return [...this.localGroups].sort((a, b) => {
        if (this.sortMethod === "size") {
          return b.count - a.count; // Sort in descending order
        }
        if (this.sortMethod === "inconsistency") {
          return (
            this.calculateInconsistency(b) - this.calculateInconsistency(a)
          ); // Sort in descending order
        }
        return 0;
      });
    },
  },
  methods: {
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
    calculateInconsistency(group) {
      const uniqueValues = new Set(group.items.map((item) => item.value));
      return uniqueValues.size / group.items.length;
    },
    updateGroups() {
      this.updateGroupSizes();
      this.$emit("update:dimension", {
        ...this.dimension,
        groups: this.sortedLocalGroups,
      });
    },
    handleGroupClick(data) {
      console.log("Group clicked:", data);
    },
    handleStandardize(group) {
      this.handleViewInTable(group);
      const groupLabel = group.name;
      let totalFrequency = 0;
      group.items.forEach((item) => {
        totalFrequency += item.count;
      });
      group.items = [
        {
          value: groupLabel,
          count: totalFrequency,
        },
      ];
      // Step 4: Update the group cards and visualizations, including the bar chart statistics
      this.updateGroups();
    //   this.updateBarChartStatistics();
      this.$emit("standardize", this.dimension.name, group.name);
    },
    handleViewInTable(group) {
      console.log(
        "Emitting from GroupedDimension:",
        this.dimension.name,
        group.items.map((v) => v.value)
      );
      this.$emit(
        "view-in-table",
        this.dimension.name,
        group.items.map((v) => v.value)
      );
    },
    getTagClass(count, total) {
      const percentage = (count / total) * 100;
      if (percentage > 66) return "high-frequency";
      if (percentage > 33) return "medium-frequency";
      return "low-frequency";
    },
    handleNameChange(groupIndex, newName) {
      this.$emit("update-group-name", {
        dimensionName: this.dimension.name,
        groupIndex,
        newName,
      });
    },
    closeTag(group, item) {
      const index = group.items.indexOf(item);
      if (index > -1) {
        group.items.splice(index, 1);
        this.closedTags.push(item);
      }
      group.count -= item.count;
      this.updateGroups();
    },
    restoreTag(tag) {
      const index = this.closedTags.indexOf(tag);
      if (index > -1) {
        this.closedTags.splice(index, 1);

        // Add to the first group only if groups exist
        if (this.localGroups.length > 0) {
          this.localGroups[0].items.push(tag);
          this.localGroups[0].count += tag.count;
        }

        this.updateGroups();
      }
    },
    createNewGroup() {
      if (this.newGroupName.trim()) {
        this.localGroups.unshift({
          name: this.newGroupName,
          count: 0,
          items: [],
        });
        this.newGroupName = "";
        this.updateGroups();
      }
    },
    // updateGroups() {
    //     this.updateGroupSizes();
    //     this.$emit('update:dimension', {
    //         ...this.dimension,
    //         groups: this.localGroups
    //     });
    // },
    handleTagMove(evt) {
      if (evt.added) {
        const { element, newIndex } = evt.added; //eslint-disable-line
        const targetGroup = this.localGroups.find((group) =>
          group.items.includes(element)
        );
        if (targetGroup) {
          targetGroup.count += element.count;
        }
      }
      if (evt.removed) {
        const { element, oldIndex } = evt.removed; //eslint-disable-line
        const sourceGroup = this.localGroups.find((group) =>
          group.items.includes(element)
        );
        if (sourceGroup) {
          sourceGroup.count -= element.count;
        }
      }
      this.updateGroupSizes();
      this.updateGroups();
    },
    updateGroupSizes() {
      this.localGroups.forEach((group) => {
        group.count = group.items.reduce((sum, item) => sum + item.count, 0);
      });
    },
  },
  watch: {
    sortedLocalGroups: {
      handler() {
        this.updateGroups();
      },
      deep: true,
    },
    "dimension.data": {
      handler() {
        this.localGroups = this.groupDataByClusters(this.dimension.data);
      },
      deep: true,
    },
  },
};
</script>

<style scoped>
.compact-legend {
  display: flex;
  align-items: center;
  margin-bottom: 10px;
  font-size: 12px;
  background-color: #f5f7fa;
  padding: 5px 10px;
  border-radius: 4px;
}

.compact-legend .legend-label {
  margin-right: 10px;
  font-weight: bold;
}

.compact-legend .el-tag {
  margin-right: 5px;
}

.group-card {
  margin-bottom: 5px;
  padding: 10px;
  border-radius: 8px;
  border: 1px solid #d3dce6;
  background-color: #f9f9f9;
  transition: box-shadow 0.2s ease-in-out;
}

.group-card:hover {
  box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
}

.group-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding-bottom: 10px;
}

.editable-category-name {
  font-size: 14px;
  font-weight: bold;
  max-width: 300px;
}

.editable-category-name /deep/ .el-input__inner {
  font-weight: bold;
  color: #303133;
}

.editable-category-name /deep/ .el-input-group__prepend {
  background-color: transparent;
  border: none;
}

.group-header-controls {
  display: flex;
  align-items: center;
}

.group-header-controls .group-size {
  margin-right: 10px;
  font-weight: bold;
}

.group-labels {
  display: flex;
  flex-wrap: wrap;
}

.tag {
  margin-right: 6px;
  margin-bottom: 6px;
  border-radius: 5px;
  color: white;
  font-weight: bold;
  display: inline-flex;
  align-items: center;
  padding: 2px 8px 2px 10px;
  font-size: 12px;
  transition: transform 0.2s ease, box-shadow 0.2s ease;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

.tag:hover {
  transform: translateY(-1px);
  box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
}

.tag-count {
  margin-left: 6px;
  background-color: rgba(255, 255, 255, 0.25);
  padding: 1px 6px;
  border-radius: 5px;
  font-size: 10px;
  font-weight: bold;
  color: inherit;
  display: inline-block;
}

/* Improved color contrast for better readability */
.high-frequency {
  background-color: #264653;
  /* Lighter dark blue */
  border: 2px solid #163a59;
  color: white;
  font-weight: bold;
}

.medium-frequency {
  background-color: #2a9d8f;
  /* Softer teal color */
  border: 2px solid #2878ab;
  color: white;
  font-weight: bold;
}

.low-frequency {
  background-color: #a8dadc;
  /* Softer yellow for low frequency */
  border: 2px solid #6aa9b5;
  color: gray;
  /* Ensures readability */
  font-weight: bold;
}

.tag-count {
  background-color: rgba(0, 0, 0, 0.1);
}

.closed-tags {
  margin-top: 10px;
  margin-bottom: 10px;
}

.closed-tags .el-tag {
  margin-right: 5px;
  margin-bottom: 5px;
  cursor: pointer;
}

.new-group {
  margin-bottom: 10px;
}

/* Add these styles for drag and drop visual feedback */
.sortable-ghost {
  opacity: 0.5;
  background: #c8ebfb;
}

.sortable-drag {
  opacity: 0.8;
  background: #f9f9f9;
}

.group-labels {
  min-height: 30px;
  display: flex;
  flex-wrap: wrap;
  /* Ensures empty groups have a drop target */
}

.group-content {
  padding-right: 10px;
  /* Add some padding for the scrollbar */
}

.groupCardLists {
  max-height: 300px;
  overflow: auto;
}
</style>
