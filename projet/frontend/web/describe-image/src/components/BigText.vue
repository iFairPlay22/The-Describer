<template>
    <el-row class="normal-y-margin">
        <el-col :span="24">
            <hr>
            <h1 :id="data.id" class="big-title">
                <el-icon class="el-icon-d-arrow-right"/>
                {{ data.text }}
            </h1>
            <hr>
        </el-col>
    </el-row>
</template>

<script>
export default {
    name: "BigText",
    props: {
        data: {
            text: {
                type: String,
                default: ""
            },
            id: {
                type: String,
                default: ""
            }
        }
    }
}
</script>

<style>

</style>