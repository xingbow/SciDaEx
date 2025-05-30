<template>
    <el-row class="custom-row">
      <div class="row-title">Add Columns</div>
      <template v-if="!isAddingColumn">
        <el-button
          v-for="(description, columnName) in columnDict"
          :key="columnName"
          class="indented-button"
          :type="isColumnSelected(columnName) ? 'primary' : ''"
          icon="el-icon-plus"
          size="mini"
          @click="toggleColumn(columnName, description)"
        >
          {{ columnName }}
        </el-button>
        <el-button
          class="indented-button"
          type="primary"
          plain
          icon="el-icon-plus"
          size="mini"
          @click="toggleAddingColumn"
        >
          Add Custom Column
        </el-button>
        <el-row style="text-align:right">
          <el-button class="indented-button" round size="mini" icon="el-icon-check" @click="submitColumns">
            confirm
          </el-button>
        </el-row>
      </template>
      <template v-else>
        <el-input
          v-model="customColumnName"
          placeholder="Add a custom column name"
          size="normal"
          class="custom-input"
        ></el-input>
        <el-input
          type="textarea"
          :autosize="{ minRows: 4, maxRows: 6 }"
          class="custom-input"
          placeholder="Description"
          v-model="textarea"
        >
        </el-input>
        <div class="button-group">
          <el-button type="default" size="small" @click="toggleAddingColumn">Cancel</el-button>
          <el-button type="primary" size="small" @click="addCustomColumn">Create</el-button>
        </div>
      </template>
    </el-row>
  </template>
  
  <script>
  export default {
    name: "AddAdditionalColumns",
    data() {
      return {
        columnDict: {
          Summary: "Concise summary of the study",
          Results: "Key results of the study",
          Limitations: "Limitations of the study",
        },
        userColumnDict: {},
        isAddingColumn: false,
        customColumnName: "",
        textarea: "",
      };
    },
    methods: {
      toggleAddingColumn() {
        this.isAddingColumn = !this.isAddingColumn;
      },
      isColumnSelected(columnName) {
      return columnName in this.userColumnDict;
    },
      toggleColumn(columnName, description) {
        if (this.isColumnSelected(columnName)) {
          this.$delete(this.userColumnDict, columnName);
        } else {
          this.$set(this.userColumnDict, columnName, description);
        }
      },
      addCustomColumn() {
        if (this.customColumnName && this.textarea) {
          this.$set(this.columnDict, this.customColumnName, this.textarea);
          this.$set(this.userColumnDict, this.customColumnName, this.textarea);
          this.customColumnName = "";
          this.textarea = "";
          this.toggleAddingColumn();
        }
      },
      submitColumns() {
        for (let columnName in this.userColumnDict) {
        if (columnName in this.columnDict) {
          this.$delete(this.columnDict, columnName);
        }
      }
      this.$emit('updateUserColumnInput', JSON.stringify(this.userColumnDict));
      this.userColumnDict = {};
      },
    },
  };

</script>

<style scoped>
.custom-row {
    flex-direction: column;
    position: relative;
    bottom: 0;
    margin-bottom: 80px;
    margin-left: 0px;
    border: 1px solid #ddd;
    border-radius: 10px;
    padding: 5px;
}

.row-title {
    margin-left: 10px;
    margin-bottom: 15px;
    font-size: 14px;
    font-weight: bold;
}

.indented-button {
    margin-bottom: 10px;
    margin-left: 10px;
}

.custom-input {
    margin-bottom: 10px;
    margin-left: 10px;
    width: 90%;
}

.button-group {
    display: flex;
    justify-content: flex-start;
    margin-left: 10px;
}

.button-group .el-button {
    margin-right: 10px;
}
</style>
