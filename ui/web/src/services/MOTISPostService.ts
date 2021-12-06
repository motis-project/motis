import axios from 'axios'
import { App } from 'vue'
import Guess from '@/models/Guess';
import GuessResponseContent from '@/models/Guess';

var service: MOTISPostService = {
    async test(input: string, gc: number) {
        let rq = {
            destination: {
                type: "Module",
                target: "/guesser"
            },
            content_type: "StationGuesserRequest",
            content: {
                input: input,
                guess_count: gc
            }
        };
        return (await axios.post<GuessResponse>("https://europe.motis-project.de/", rq)).data.content;
    }
}

interface GuessResponse {
    destination: {
        type: string,
        target: string
    },
    content_type: string,
    content: GuessResponseContent,
    id: number
}

interface MOTISPostService {
    test(input: string, gc: number): Promise<GuessResponseContent>
}


















declare module '@vue/runtime-core' {
    interface ComponentCustomProperties {
        $postService: MOTISPostService
    }
}

export default {
    install: (app: App, options: string) => {
        app.config.globalProperties.$postService = service;
    }
}