<template>
    <header>
        <el-menu class="header" mode="horizontal">
            <el-menu-item class="text-important" style="margin-top: 1px">
                {{ data.name }}
            </el-menu-item>
            <el-menu-item v-for="({ name, id }, i) in data.links" :key="i" class="xs-disable">
                <el-link type="primary"> 
                    <div @click="goToLink(id)">
                        {{ name }} 
                    </div>
                </el-link>
            </el-menu-item>
        </el-menu>
    </header>
</template>

<script>
export default {
    name: "MainHeader",
    props: {
        data: {
            name: {
                type: String,
                default: ""
            },
            links: {
                type: Array,
                default: []
            },
        }
    },
    methods: {
        goToLink(id) {
            const el = document.getElementById(id);
            el.scrollIntoView({ behavior: "smooth" });
        }
    },
}
</script>

<style>

</style>