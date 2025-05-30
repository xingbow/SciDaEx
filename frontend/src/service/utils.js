/* global $ */
import * as dfd from "danfojs"
import {TabulatorFull as Tabulator} from 'tabulator-tables'; //import Tabulator library

// Utility function to escape RegExp special characters
function escapeRegExp(string) {
    return string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

function highlightcells(cell){
    if(cell.getValue()  == "Empty" || (cell.getValue()  == "empty") ){
        $(cell.getElement()).css({"background-color":"rgba(255, 0, 0, 0.3)", });
      }
      else{
        $(cell.getElement()).css({"background-color":"transparent", });
      }
}

function isEmpty(val){
    return (val === undefined || val == null || val.length <= 0) ? true : false;
}

function nameFilter(data, filterParams){
    let filenames = filterParams.map((d) => d.filename)
    return filenames.includes(data.pdf_file.split(".")[0])
}

function joinArrays(arr1, arr2, key, joinType='inner') {
    // let joined = [];

    // Handle cases where arr1 or arr2 are empty
    if (arr1.length === 0) {
        return joinType === 'outer' ? arr2 : [];
    }
    if (arr2.length === 0) {
        return joinType === 'outer' ? arr1 : (joinType === 'left' ? arr1 : []);
    }

    let df1 = new dfd.DataFrame(arr1);
    let df2 = new dfd.DataFrame(arr2);
    let df = dfd.merge({ "left": df1, "right": df2, "on": key, how: joinType})
    df.fillNa("",{inplace: true});
    return dfd.toJSON(df, {format: "column"})
}


function buildTable(divid, options = {}) {
    
    // Set default values for options
    const {
        contentEdittable = false,
        colTitleEditable = false,
        pagination = true,
        paginationSize = 9,
        paginationCounter = "rows"
    } = options;
    
    // function implementation using these options
    return new Tabulator("#" + divid, {
        pagination: pagination,
        paginationSize: paginationSize,
        paginationCounter: paginationCounter,
        autoColumns: true,
        autoColumnsDefinitions: function (definitions) {
            definitions.forEach((column) => {
                column.editor = contentEdittable;
                column.editableTitle = colTitleEditable;
            });
            return definitions;
        },
    });
}


function formatCitationAPA(paper) {
    let authors = paper.Author.split(', ').map(author => {
        let parts = author.split(' ');
        let lastName = parts.pop();
        let initials = parts.map(name => name[0] + '.').join(' ');
        return `${lastName}, ${initials}`;
    }).join(', ');

    let title = paper.Title;
    let journal = paper.Journal_or_Conference;
    let volume = paper.Volume;
    let pages = `${paper.Page.start}-${paper.Page.end}`;
    let year = paper.Year;
    let doi = paper.DOI;

    return `${authors} (${year}). ${title}. ${journal}, ${volume}, ${pages}. ${doi}`;
}

function isAlert(dataObj, th = .25){
    let total_records = 0, alert_records = 0;
    Object.keys(dataObj).map((key)=>{
        if(key!="pdf_file"){
            total_records += 1;
            if(isEmpty(dataObj[key]) || String(dataObj[key]).trim().toLowerCase() == "empty"){
              alert_records += 1;
            }
        }
    });
    if(alert_records/total_records >= th){
        return true;
    }else{
        return false;
    }
}

export default {
    isEmpty,
    isAlert,
    joinArrays,
    nameFilter,
    buildTable,
    formatCitationAPA,
    highlightcells,
    escapeRegExp,
}