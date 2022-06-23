<template>
  <el-image
    class="qr_code"
    :src="data.url"
    fit="cover"
    />
</template>

<script>
export default {
    name: "AbsoluteQrCode",
    props: {
        data: {
            url: {
                type: String,
                default: ""
            }
        }
    }
}
</script>