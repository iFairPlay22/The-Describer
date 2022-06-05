<template>
  <header>
    <el-menu class="el-menu-demo" mode="horizontal">
        <el-menu-item class="text-important" style="margin-top: 1px">
            The Describers
        </el-menu-item>
        <el-menu-item v-for="({ name, url }, i) in links" :key="i">
            <el-link type="primary" :href="url" target="_blank"> {{ name }} </el-link>
        </el-menu-item>
    </el-menu>
</header>
</template>

<script>
export default {
    name: "MainHeader",
    data() {
        return {
            links: [
                {
                    name: "Processing Center",
                    url: "/"
                },
                {
                    name: "Orders",
                    url: "https://www.ele.me"
                }
            ]
        }
    },
}
</script>

<style>

</style>