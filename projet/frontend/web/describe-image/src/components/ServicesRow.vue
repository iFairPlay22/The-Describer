<template>
  <el-row class="normal-y-margin">
        <el-col :span="24">
            <div class="services-container">
                <el-card class="service-card" v-for="({ avatar, title, description, button, callback }, i) in data.services" :key="i">
                    <div class="service-card-content">
                        <el-avatar :src="avatar" :size="100"/>
                        <div class="title">{{ title }}</div>
                        <div class="text">{{ description }}</div>
                    </div>
                    <div class="service-card-buttons">
                        <el-button type="primary" class="very-small-margin" @click="callback"> 
                            <el-icon class="el-icon-download"/>
                            {{ button }} 
                        </el-button>
                    </div>
                </el-card>
            </div>
        </el-col>
    </el-row>
</template>

<script>
export default {
    name: "ServicesRow",
    props: {
        data: {
            services: { type: Array, default: [] },
      }
    }
}
</script>

<style>
    .card-with-image-bg {
        position: relative;
    }
    .card-with-image-bg .card-content {
        position: relative;
        z-index: 2;
    }
    .card-with-image-bg .image-bg {
        position: absolute;
        left: 0;
        top: 0;
        width: 100%;
        height: auto;
        opacity: 0.9;
    }

</style>