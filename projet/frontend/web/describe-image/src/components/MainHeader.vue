<template>
  <header>
    <el-menu class="el-menu-demo" mode="horizontal">
        <el-menu-item class="text-important" style="margin-top: 1px">
            The Describers
        </el-menu-item>
        <el-menu-item v-for="({ name, id }, i) in links" :key="i" class="xs-disable">
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
    data() {
        return {
            links: [
                {
                    name: "Origine",
                    id: "what"
                },
                {
                    name: "L'IA",
                    id: "ai"
                }
            ]
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