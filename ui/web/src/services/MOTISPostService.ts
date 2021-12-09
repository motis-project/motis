import axios from 'axios'
import { App } from 'vue'
import { StationGuessResponseContent } from '../models/StationGuess';
import { AddressGuessResponseContent } from '../models/AddressGuess';
import Trip from '../models/Trip';
import TripResponseContent from '../models/TripResponseContent';


var service: MOTISPostService = {
  async getStationGuessResponse(input: string, gc: number) {
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
    return (await axios.post<StationGuessResponse>("https://europe.motis-project.de/", rq)).data.content;
  },
  async getAddressGuessResponse(input: string) {
    let rq = {
      destination: {
        type: "Module",
        target: "/address"
      },
      content_type: "AddressRequest",
      content: {
        input: input
      }
    };
    return (await axios.post<AddressGuessResponse>("https://europe.motis-project.de/", rq)).data.content;
  },
  async getTripResponce(trip: Trip) {
    let rq = {
      destination: {
        type: "Module",
        target: "/trip_to_connection"
      },
      content_type: "TripId",
      content: trip
    };
    return (await axios.post<TripResponce>("https://europe.motis-project.de/", rq)).data.content;
  }
}

interface StationGuessResponse {
  destination: {
    type: string,
    target: string
  },
  content_type: string,
  content: StationGuessResponseContent,
  id: number
}

interface AddressGuessResponse {
  destination: {
    target: string
  },
  content_type: string,
  content: AddressGuessResponseContent,
  id: 1
}

interface TripResponce {
  destination: {
    type: string,
    target: string
  },
  content_type: string,
  content: TripResponseContent,
  id: number
}


interface MOTISPostService {
  getStationGuessResponse(input: string, gc: number): Promise<StationGuessResponseContent>
  getAddressGuessResponse(input: string): Promise<AddressGuessResponseContent>
  getTripResponce(input: Trip) : Promise<TripResponseContent>
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