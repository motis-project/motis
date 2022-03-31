import axios from "axios";
const mockedAxios = axios as jest.Mocked<typeof axios>

export function getRandomInt(min: number, max: number): number {
  max++;
  return Math.floor(Math.random() * (max - min)) + min;
}

export function mockNextAxiosPost(data: unknown): void {
  mockedAxios.post.mockResolvedValueOnce({data: data})
}
