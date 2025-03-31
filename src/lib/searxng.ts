/**
 * SearXNG 검색 엔진을 사용하여 웹 검색을 수행하는 모듈
 */
import axios from 'axios';
import { getSearxngApiEndpoint } from './config';

/**
 * SearXNG 검색 옵션을 정의하는 인터페이스
 * @interface SearxngSearchOptions
 * @property {string[]} [categories] - 검색할 카테고리 목록 (예: 'general', 'images', 'news')
 * @property {string[]} [engines] - 사용할 검색 엔진 목록 (예: 'google', 'bing', 'duckduckgo')
 * @property {string} [language] - 검색 결과의 언어 (예: 'en', 'ko', 'ja')
 * @property {number} [pageno] - 결과 페이지 번호
 */
interface SearxngSearchOptions {
  categories?: string[];
  engines?: string[];
  language?: string;
  pageno?: number;
}

/**
 * SearXNG 검색 결과 항목을 정의하는 인터페이스
 * @interface SearxngSearchResult
 * @property {string} title - 검색 결과의 제목
 * @property {string} url - 검색 결과의 URL
 * @property {string} [img_src] - 이미지 검색 결과의 이미지 URL
 * @property {string} [thumbnail_src] - 썸네일 이미지 URL
 * @property {string} [thumbnail] - 대체 썸네일 URL
 * @property {string} [content] - 검색 결과의 내용 요약 또는 스니펫
 * @property {string} [author] - 콘텐츠 작성자 (뉴스, 블로그 등에서 사용)
 * @property {string} [iframe_src] - iframe 소스 URL (비디오 등에서 사용)
 */
interface SearxngSearchResult {
  title: string;
  url: string;
  img_src?: string;
  thumbnail_src?: string;
  thumbnail?: string;
  content?: string;
  author?: string;
  iframe_src?: string;
}

/**
 * SearXNG 검색 엔진을 사용하여 웹 검색을 수행하는 함수
 * 
 * 이 함수는 제공된 쿼리와 옵션을 사용하여 SearXNG API에 검색 요청을 보내고
 * 검색 결과와 제안 쿼리를 포함한 응답을 반환합니다.
 * 
 * @param {string} query - 검색할 쿼리 문자열
 * @param {SearxngSearchOptions} [opts] - 선택적 검색 옵션
 * @returns {Promise<{results: SearxngSearchResult[], suggestions: string[]}>} 검색 결과와 제안 쿼리
 * 
 * @example
 * // 기본 검색
 * const { results, suggestions } = await searchSearxng('TypeScript tutorial');
 * 
 * @example
 * // 옵션을 사용한 검색
 * const { results } = await searchSearxng('AI news', {
 *   engines: ['google', 'bing'],
 *   language: 'en',
 *   categories: ['news'],
 *   pageno: 1
 * });
 */
export const searchSearxng = async (
  query: string,
  opts?: SearxngSearchOptions,
) => {
  // SearXNG API 엔드포인트 URL 가져오기
  const searxngURL = getSearxngApiEndpoint();

  // 검색 URL 생성 및 JSON 형식 결과 요청
  const url = new URL(`${searxngURL}/search?format=json`);
  // 검색 쿼리 추가
  url.searchParams.append('q', query);

  // 검색 옵션이 제공된 경우 URL 파라미터에 추가
  if (opts) {
    Object.keys(opts).forEach((key) => {
      const value = opts[key as keyof SearxngSearchOptions];
      // 배열 타입의 값은 쉼표로 구분하여 단일 문자열로 변환
      if (Array.isArray(value)) {
        url.searchParams.append(key, value.join(','));
        return;
      }
      // 단일 값은 그대로 추가
      url.searchParams.append(key, value as string);
    });
  }

  // Axios를 사용하여 SearXNG API에 GET 요청 보내기
  const res = await axios.get(url.toString());

  // 응답에서 검색 결과와 제안 쿼리 추출
  const results: SearxngSearchResult[] = res.data.results;
  const suggestions: string[] = res.data.suggestions;

  // 검색 결과와 제안 쿼리 반환
  return { results, suggestions };
};