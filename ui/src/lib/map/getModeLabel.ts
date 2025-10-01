import type { Mode } from "$lib/api/openapi"

export const getModeLabel = (mode: Mode) : string => {
    switch(mode) {
        case 'BUS':
        case 'FERRY':	
        case 'TRAM':
        case 'COACH':
        case 'AIRPLANE':
        case 'AERIAL_LIFT':
            return 'Platform'
        default:
            return 'Track'
    }
}