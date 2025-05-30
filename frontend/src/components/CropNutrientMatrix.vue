<template>
    <div class="crop-nutrient-matrix">
      <h3>Crop Nutrient Matrix</h3>
      <el-table :data="tableData" border style="width: 100%" size="mini">
        <el-table-column prop="crop" label="Crop" width="180"></el-table-column>
        <el-table-column v-for="nutrient in nutrients" :key="nutrient" :label="nutrient">
          <template slot-scope="scope">
            {{ getCellValue(scope.row.crop, nutrient) }}
          </template>
        </el-table-column>
      </el-table>
    </div>
  </template>
  
  <script>
  export default {
    name: 'CropNutrientMatrix',
    props: {
      data: {
        type: Array,
        required: true
      }
    },
    computed: {
      crops() {
        return [...new Set(this.data.map(group => group.name.split(' in ')[1]))];
      },
      nutrients() {
        return [...new Set(this.data.map(group => group.name.split(' in ')[0]))];
      },
      tableData() {
        return this.crops.map(crop => ({
          crop,
          ...Object.fromEntries(this.nutrients.map(nutrient => [nutrient, this.getCellValue(crop, nutrient)]))
        }));
      }
    },
    methods: {
      getCellValue(crop, nutrient) {
        const cell = this.data.find(group => group.name === `${nutrient} in ${crop}`);
        return cell ? `${cell.values.length} / ${cell.size}` : '-';
      }
    }
  };
  </script>
  
  <style scoped>
  .crop-nutrient-matrix {
    margin-bottom: 20px;
  }
  </style>