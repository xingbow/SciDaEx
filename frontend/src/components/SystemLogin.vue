<template>
    <el-dialog
      title="System Login"
      :show-close="false"
      :visible.sync="localDialogFormVisible"
    >
      <el-form :model="localForm" style="text-align: left !important;">
        <el-form-item label="First Name" :label-width="formLabelWidth">
          <el-input v-model="localForm.name" autocomplete="off"></el-input>
        </el-form-item>
        <el-form-item label="User ID" :label-width="formLabelWidth">
          <el-select v-model="localForm.userid" placeholder="Please select a user ID">
            <el-option
              v-for="n in 12"
              :key="n"
              :label="String(n)"
              :value="n"
            ></el-option>
          </el-select>
        </el-form-item>
        <el-form-item label="Password" :label-width="formLabelWidth">
          <el-input
            placeholder="Please input password"
            v-model="localPasswd"
            show-password
          ></el-input>
        </el-form-item>
      </el-form>
      <span slot="footer" class="dialog-footer">
        <el-button type="primary" @click="confirmLogin">Confirm</el-button>
      </span>
    </el-dialog>
  </template>
  
  <script>

  import utils from '../service/utils.js';
  import service from "../service/service.js";
  export default {
    props: {
      dialogFormVisible: {
        type: Boolean,
        required: true
      },
      form: {
        type: Object,
        required: true
      },
      formLabelWidth: {
        type: String,
        default: '120px'
      },
      passwd: {
        type: String,
        required: true
      }
    },
    data() {
      return {
        localDialogFormVisible: this.dialogFormVisible,
        localForm: { ...this.form },
        localPasswd: this.passwd
      };
    },
    watch: {
      dialogFormVisible(val) {
        this.localDialogFormVisible = val;
      },
      localDialogFormVisible(val) {
        this.$emit('update:dialogFormVisible', val);
      },
      form(val) {
        this.localForm = { ...val };
      },
      passwd(val) {
        this.localPasswd = val;
      }
    },
    methods: {
      confirmLogin() {
        if(utils.isEmpty(this.localForm.userid)){
          alert("Please enter your user id");
        }else{
          if(this.localPasswd == service.loginpassword){
            this.localDialogFormVisible = false;
          }else{
            alert("Please enter the correct password");
            this.localDialogFormVisible = true;
          }
        }
      },
    }
  };
  </script>
  
  <style scoped>
  /* Add your component-specific styles here */
  </style>
  