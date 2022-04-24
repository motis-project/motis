export default {
  methods: {
    convertClassZ(classZ: number): string {
        switch (classZ) {
        case 1:
        case 2:
        case 4:
        case 5:
        case 6:
          return "train";
        case 0:
          return "plane";
        case 7:
          return "sbahn";
        case 8:
          return "ubahn";
        case 9:
          return "tram";
        case 11:
          return "ship";
        case 3:
        default:
          return "bus";
        }
    }
  }
}
