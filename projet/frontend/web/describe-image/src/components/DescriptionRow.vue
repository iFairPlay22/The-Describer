<template>
    <el-row class="normal-y-margin">
        <el-col :span="24" :md="12" class="merge-container-left">
            <el-row style="margin: 20px 0">
                <el-col :span="24">
                    <el-card class="without-borders">
                        <div class="title">{{ data.title }}</div>
                        <div class="text">{{ data.text }}</div>
                    </el-card>
                </el-col>
            </el-row>
            <el-row>
                <el-col :span="24">
                    <el-row>
                        <el-col :xs="24" :sm="12">
                            <el-card class="rounded-borders-except-bl small-margin merge-cards-left">
                                <div class="title">{{ data.groups[0].title }}</div>
                                <div class="text">{{ data.groups[0].text }}</div>
                            </el-card>
                        </el-col>
                        <el-col :xs="24" :sm="12">
                            <el-card class="rounded-borders-except-tr small-margin merge-cards-right">
                                <div class="title">{{ data.groups[1].title }}</div>
                                <div class="text">{{ data.groups[1].text }}</div>
                            </el-card>
                        </el-col>
                    </el-row>
                </el-col>
            </el-row>
        </el-col>
        <el-col :xs="24" :sm="12" class="merge-container-right">
            <el-image
                class="big-home-image"
                :src="data.image"
                fit="cover">
            </el-image>
        </el-col>
    </el-row>
</template>

<script>
export default {
  name: "DescriptionRow",
  props: {
      data: {
          title: {
            type: String,
            default: "",
         },
          text: {
            type: String,
            default: "",
         },
          image: {
            type: String,
            default: "",
         },
          groups: {
            type: Array,
            default: [ 
                { title: "", text: "" },
                { title: "", text: "" }
            ],
         },
      }
  }
}
</script>

<style>

</style>