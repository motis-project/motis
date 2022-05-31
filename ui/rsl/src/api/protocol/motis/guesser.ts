// generated file - do not modify - run update-protocol to update
import { Station } from "@/api/protocol/motis";

// guesser/StationGuesserRequest.fbs
export interface StationGuesserRequest {
  guess_count: number; // default: 8
  input: string;
}

// guesser/StationGuesserResponse.fbs
export interface StationGuesserResponse {
  guesses: Station[];
}
