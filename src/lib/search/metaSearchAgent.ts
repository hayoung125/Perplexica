import { ChatOpenAI } from '@langchain/openai';
import type { BaseChatModel } from '@langchain/core/language_models/chat_models';
import type { Embeddings } from '@langchain/core/embeddings';
import {
  ChatPromptTemplate,
  MessagesPlaceholder,
  PromptTemplate,
} from '@langchain/core/prompts';
import {
  RunnableLambda,
  RunnableMap,
  RunnableSequence,
} from '@langchain/core/runnables';
import { BaseMessage } from '@langchain/core/messages';
import { StringOutputParser } from '@langchain/core/output_parsers';
import LineListOutputParser from '../outputParsers/listLineOutputParser';
import LineOutputParser from '../outputParsers/lineOutputParser';
import { getDocumentsFromLinks } from '../utils/documents';
import { Document } from 'langchain/document';
import { searchSearxng } from '../searxng';
import path from 'node:path';
import fs from 'node:fs';
import computeSimilarity from '../utils/computeSimilarity';
import formatChatHistoryAsString from '../utils/formatHistory';
import eventEmitter from 'events';
import { StreamEvent } from '@langchain/core/tracers/log_stream';

export interface MetaSearchAgentType {
  searchAndAnswer: (
    message: string,
    history: BaseMessage[],
    llm: BaseChatModel,
    embeddings: Embeddings,
    optimizationMode: 'speed' | 'balanced' | 'quality',
    fileIds: string[],
  ) => Promise<eventEmitter>;
}

interface Config {
  searchWeb: boolean;
  rerank: boolean;
  summarizer: boolean;
  rerankThreshold: number;
  queryGeneratorPrompt: string;
  responsePrompt: string;
  activeEngines: string[];
}

type BasicChainInput = {
  chat_history: BaseMessage[];
  query: string;
};

class MetaSearchAgent implements MetaSearchAgentType {
  private config: Config;
  private strParser = new StringOutputParser();

  constructor(config: Config) {
    this.config = config;
  }

  /**
 * 검색 리트리버 체인을 생성하는 메서드
 * 이 메서드는 사용자 질의를 처리하고, 웹 검색을 수행하며, 관련 문서를 가져오고 요약하는 처리 파이프라인을 만듭니다.
 */
  private async createSearchRetrieverChain(llm: BaseChatModel) {
    // 언어 모델의 온도를 0으로 설정 (결정적인 출력을 위해)
    (llm as unknown as ChatOpenAI).temperature = 0;

    // RunnableSequence는 여러 단계를 순차적으로 처리하는 파이프라인을 정의합니다
    return RunnableSequence.from([
      // 1단계: 사용자 질의를 받아 프롬프트 템플릿 적용
      PromptTemplate.fromTemplate(this.config.queryGeneratorPrompt),
      // 2단계: 언어 모델로 프롬프트 처리
      llm,
      // 3단계: 언어 모델의 출력을 문자열로 파싱
      this.strParser,
      // 4단계: 파싱된 출력을 처리하는 람다 함수
      RunnableLambda.from(async (input: string) => {
        // 출력에서 링크 목록 추출을 위한 파서
        const linksOutputParser = new LineListOutputParser({
          key: 'links',
        });

        // 출력에서 질문 추출을 위한 파서
        const questionOutputParser = new LineOutputParser({
          key: 'question',
        });

        // 출력에서 링크 목록 추출
        const links = await linksOutputParser.parse(input);
        // config.summarizer가 활성화된 경우에만 question 파서 사용, 아니면 전체 입력 사용
        let question = this.config.summarizer
          ? await questionOutputParser.parse(input)
          : input;

        // 'not_needed'인 경우 검색이 필요 없음을 나타냄 (빈 결과 반환)
        if (question === 'not_needed') {
          return { query: '', docs: [] };
        }

        // 링크가 제공된 경우 (URL 기반 검색)
        if (links.length > 0) {
          // 질문이 비어있으면 요약 모드로 설정
          if (question.length === 0) {
            question = 'summarize';
          }

          // 문서 저장을 위한 배열 초기화
          let docs: Document[] = [];

          // 제공된 링크에서 문서 가져오기
          const linkDocs = await getDocumentsFromLinks({ links });

          // 문서 그룹화를 위한 배열 초기화 (URL별 그룹화)
          const docGroups: Document[] = [];

          // 링크에서 가져온 문서들을 URL 기준으로 그룹화
          linkDocs.map((doc) => {
            // 동일한 URL의 문서가 이미 있는지 확인 (최대 10개 제한)
            const URLDocExists = docGroups.find(
              (d) =>
                d.metadata.url === doc.metadata.url &&
                d.metadata.totalDocs < 10,
            );

            // 해당 URL의 문서가 아직 없으면 새 그룹 생성
            if (!URLDocExists) {
              docGroups.push({
                ...doc,
                metadata: {
                  ...doc.metadata,
                  totalDocs: 1,
                },
              });
            }

            // 같은 URL의 문서를 다시 찾아 인덱스 확인
            const docIndex = docGroups.findIndex(
              (d) =>
                d.metadata.url === doc.metadata.url &&
                d.metadata.totalDocs < 10,
            );

            // 인덱스가 존재하면 해당 문서 그룹에 내용 추가
            if (docIndex !== -1) {
              docGroups[docIndex].pageContent =
                docGroups[docIndex].pageContent + `\n\n` + doc.pageContent;
              docGroups[docIndex].metadata.totalDocs += 1;
            }
          });

          // 각 문서 그룹을 병렬로 처리하여 요약
          await Promise.all(
            docGroups.map(async (doc) => {
              // 각 문서에 대해 LLM을 사용하여 요약 수행
              const res = await llm.invoke(`
            You are a web search summarizer, tasked with summarizing a piece of text retrieved from a web search. Your job is to summarize the 
            text into a detailed, 2-4 paragraph explanation that captures the main ideas and provides a comprehensive answer to the query.
            If the query is \"summarize\", you should provide a detailed summary of the text. If the query is a specific question, you should answer it in the summary.
            
            - **Journalistic tone**: The summary should sound professional and journalistic, not too casual or vague.
            - **Thorough and detailed**: Ensure that every key point from the text is captured and that the summary directly answers the query.
            - **Not too lengthy, but detailed**: The summary should be informative but not excessively long. Focus on providing detailed information in a concise format.

            The text will be shared inside the \`text\` XML tag, and the query inside the \`query\` XML tag.

            <example>
            1. \`<text>
            Docker is a set of platform-as-a-service products that use OS-level virtualization to deliver software in packages called containers. 
            It was first released in 2013 and is developed by Docker, Inc. Docker is designed to make it easier to create, deploy, and run applications 
            by using containers.
            </text>

            <query>
            What is Docker and how does it work?
            </query>

            Response:
            Docker is a revolutionary platform-as-a-service product developed by Docker, Inc., that uses container technology to make application 
            deployment more efficient. It allows developers to package their software with all necessary dependencies, making it easier to run in 
            any environment. Released in 2013, Docker has transformed the way applications are built, deployed, and managed.
            \`
            2. \`<text>
            The theory of relativity, or simply relativity, encompasses two interrelated theories of Albert Einstein: special relativity and general
            relativity. However, the word "relativity" is sometimes used in reference to Galilean invariance. The term "theory of relativity" was based
            on the expression "relative theory" used by Max Planck in 1906. The theory of relativity usually encompasses two interrelated theories by
            Albert Einstein: special relativity and general relativity. Special relativity applies to all physical phenomena in the absence of gravity.
            General relativity explains the law of gravitation and its relation to other forces of nature. It applies to the cosmological and astrophysical
            realm, including astronomy.
            </text>

            <query>
            summarize
            </query>

            Response:
            The theory of relativity, developed by Albert Einstein, encompasses two main theories: special relativity and general relativity. Special
            relativity applies to all physical phenomena in the absence of gravity, while general relativity explains the law of gravitation and its
            relation to other forces of nature. The theory of relativity is based on the concept of "relative theory," as introduced by Max Planck in
            1906. It is a fundamental theory in physics that has revolutionized our understanding of the universe.
            \`
            </example>

            Everything below is the actual data you will be working with. Good luck!

            <query>
            ${question}
            </query>

            <text>
            ${doc.pageContent}
            </text>

            Make sure to answer the query in the summary.
          `);

              // 요약된 내용을 새 Document 객체로 생성
              const document = new Document({
                pageContent: res.content as string,
                metadata: {
                  title: doc.metadata.title,
                  url: doc.metadata.url,
                },
              });

              // 요약 문서를 결과 배열에 추가
              docs.push(document);
            }),
          );

          // 질문과 요약된 문서들 반환
          return { query: question, docs: docs };
        } else {
          // 링크가 없는 경우 (일반 웹 검색)
          // "think" 태그 제거 (내부 사고 과정 제거)
          question = question.replace(/<think>.*?<\/think>/g, '');

          // SearXNG 검색 엔진을 사용하여 질문에 대한 검색 수행
          const res = await searchSearxng(question, {
            language: 'en',
            engines: this.config.activeEngines,
          });

          // 검색 결과를 Document 객체 배열로 변환
          const documents = res.results.map(
            (result) =>
              new Document({
                pageContent:
                  result.content ||
                  (this.config.activeEngines.includes('youtube')
                    ? result.title
                    : '') /* Todo: Implement transcript grabbing using Youtubei (source: https://www.npmjs.com/package/youtubei) */,
                metadata: {
                  title: result.title,
                  url: result.url,
                  ...(result.img_src && { img_src: result.img_src }),
                },
              }),
          );

          // 질문과 검색된 문서들 반환
          return { query: question, docs: documents };
        }
      }),
    ]);
  }

  private async createAnsweringChain(
    llm: BaseChatModel,
    fileIds: string[],
    embeddings: Embeddings,
    optimizationMode: 'speed' | 'balanced' | 'quality',
  ) {
    return RunnableSequence.from([
      RunnableMap.from({
        query: (input: BasicChainInput) => input.query,
        chat_history: (input: BasicChainInput) => input.chat_history,
        date: () => new Date().toISOString(),
        context: RunnableLambda.from(async (input: BasicChainInput) => {
          const processedHistory = formatChatHistoryAsString(
            input.chat_history,
          );

          let docs: Document[] | null = null;
          let query = input.query;

          if (this.config.searchWeb) {
            const searchRetrieverChain =
              await this.createSearchRetrieverChain(llm);

            const searchRetrieverResult = await searchRetrieverChain.invoke({
              chat_history: processedHistory,
              query,
            });

            query = searchRetrieverResult.query;
            docs = searchRetrieverResult.docs;
          }

          const sortedDocs = await this.rerankDocs(
            query,
            docs ?? [],
            fileIds,
            embeddings,
            optimizationMode,
          );
          return sortedDocs;
        })
          .withConfig({
            runName: 'FinalSourceRetriever',
          })
          .pipe(this.processDocs),
      }),
      ChatPromptTemplate.fromMessages([
        ['system', this.config.responsePrompt],
        new MessagesPlaceholder('chat_history'),
        ['user', '{query}'],
      ]),
      llm,
      this.strParser,
    ]).withConfig({
      runName: 'FinalResponseGenerator',
    });
  }

  /**
 * 문서를 쿼리와의 관련성에 따라 재정렬하는 함수
 * 
 * 이 함수는 검색된 문서와 파일 ID를 바탕으로 쿼리와의 유사도를 계산하고,
 * 가장 관련성이 높은 문서들을 반환합니다. 최적화 모드에 따라 처리 속도와 정확도의 
 * 균형을 조절할 수 있습니다.
 * 
 * @param query - 사용자의 검색 쿼리 문자열
 * @param docs - 검색된 문서 배열
 * @param fileIds - 참조할 파일 ID 배열
 * @param embeddings - 텍스트를 벡터로 변환하는 임베딩 모델
 * @param optimizationMode - 처리 최적화 모드 ('speed': 속도 우선, 'balanced': 균형, 'quality': 품질 우선)
 * @returns 재정렬된 문서 배열
 */
private async rerankDocs(
    query: string,
    docs: Document[],
    fileIds: string[],
    embeddings: Embeddings,
    optimizationMode: 'speed' | 'balanced' | 'quality',
  ) {
    // 문서와 파일이 모두 없는 경우 빈 문서 배열 반환
    if (docs.length === 0 && fileIds.length === 0) {
      return docs;
    }

    // 파일 ID를 사용하여 파일 데이터 로드 및 처리
    const filesData = fileIds
      .map((file) => {
        // 파일 경로 구성
        const filePath = path.join(process.cwd(), 'uploads', file);

        // 추출된 내용과 임베딩 파일 경로
        const contentPath = filePath + '-extracted.json';
        const embeddingsPath = filePath + '-embeddings.json';

        // 파일 내용과 임베딩 데이터 로드
        const content = JSON.parse(fs.readFileSync(contentPath, 'utf8'));
        const embeddings = JSON.parse(fs.readFileSync(embeddingsPath, 'utf8'));

        // 파일 내용과 임베딩을 결합하여 검색 가능한 객체 배열 생성
        const fileSimilaritySearchObject = content.contents.map(
          (c: string, i: number) => {
            return {
              fileName: content.title,
              content: c,
              embeddings: embeddings.embeddings[i],
            };
          },
        );

        return fileSimilaritySearchObject;
      })
      .flat(); // 모든 파일 데이터를 하나의 배열로 평탄화

    // 쿼리가 'summarize'인 경우 추가 처리 없이 상위 15개 문서 반환
    if (query.toLocaleLowerCase() === 'summarize') {
      return docs.slice(0, 15);
    }

    // 실제 내용이 있는 문서만 필터링
    const docsWithContent = docs.filter(
      (doc) => doc.pageContent && doc.pageContent.length > 0,
    );

    // 속도 우선 모드 또는 재정렬 기능이 비활성화된 경우
    if (optimizationMode === 'speed' || this.config.rerank === false) {
      // 참조할 파일이 있는 경우
      if (filesData.length > 0) {
        // 쿼리를 임베딩 벡터로 변환
        const [queryEmbedding] = await Promise.all([
          embeddings.embedQuery(query),
        ]);

        // 파일 데이터를 Document 객체로 변환
        const fileDocs = filesData.map((fileData) => {
          return new Document({
            pageContent: fileData.content,
            metadata: {
              title: fileData.fileName,
              url: `File`,
            },
          });
        });

        // 파일 데이터의 임베딩과 쿼리 임베딩 간의 유사도 계산
        const similarity = filesData.map((fileData, i) => {
          const sim = computeSimilarity(queryEmbedding, fileData.embeddings);

          return {
            index: i,
            similarity: sim,
          };
        });

        // 유사도에 따라 문서 정렬 및 필터링
        let sortedDocs = similarity
          .filter(
            (sim) => sim.similarity > (this.config.rerankThreshold ?? 0.3),
          ) // 임계값보다 높은 유사도를 가진 문서만 선택
          .sort((a, b) => b.similarity - a.similarity) // 유사도 내림차순 정렬
          .slice(0, 15) // 상위 15개 선택
          .map((sim) => fileDocs[sim.index]); // 해당 인덱스의 문서 매핑

        // 웹 검색 결과가 있으면 파일 문서 수를 제한
        sortedDocs =
          docsWithContent.length > 0 ? sortedDocs.slice(0, 8) : sortedDocs;

        // 파일 문서와 웹 검색 결과 결합 (총 15개 문서까지)
        return [
          ...sortedDocs,
          ...docsWithContent.slice(0, 15 - sortedDocs.length),
        ];
      } else {
        // 파일이 없는 경우 웹 검색 결과만 반환 (최대 15개)
        return docsWithContent.slice(0, 15);
      }
    } 
    // 균형 모드인 경우
    else if (optimizationMode === 'balanced') {
      // 문서 내용과 쿼리를 동시에 임베딩 벡터로 변환
      const [docEmbeddings, queryEmbedding] = await Promise.all([
        embeddings.embedDocuments(
          docsWithContent.map((doc) => doc.pageContent),
        ),
        embeddings.embedQuery(query),
      ]);

      // 파일 데이터를 Document 객체로 변환하여 문서 배열에 추가
      docsWithContent.push(
        ...filesData.map((fileData) => {
          return new Document({
            pageContent: fileData.content,
            metadata: {
              title: fileData.fileName,
              url: `File`,
            },
          });
        }),
      );

      // 파일 임베딩을 문서 임베딩 배열에 추가
      docEmbeddings.push(...filesData.map((fileData) => fileData.embeddings));

      // 모든 문서의 임베딩과 쿼리 임베딩 간의 유사도 계산
      const similarity = docEmbeddings.map((docEmbedding, i) => {
        const sim = computeSimilarity(queryEmbedding, docEmbedding);

        return {
          index: i,
          similarity: sim,
        };
      });

      // 유사도에 따라 문서 정렬 및 필터링
      const sortedDocs = similarity
        .filter((sim) => sim.similarity > (this.config.rerankThreshold ?? 0.3)) // 임계값보다 높은 유사도를 가진 문서만 선택
        .sort((a, b) => b.similarity - a.similarity) // 유사도 내림차순 정렬
        .slice(0, 15) // 상위 15개 선택
        .map((sim) => docsWithContent[sim.index]); // 해당 인덱스의 문서 매핑

      return sortedDocs;
    }

    // 지원되지 않는 최적화 모드의 경우 빈 배열 반환
    return [];
  }

  /**
   * 문서 배열을 텍스트 형식으로 처리하는 함수
   * 
   * 이 함수는 Document 객체 배열을 받아 번호가 매겨진 텍스트 문자열로 변환합니다.
   * 각 문서는 순차적인 번호, 문서 제목, 그리고 문서 내용이 포함된 형태로 변환됩니다.
   * 모든 문서가 하나의 문자열로 결합되어 반환됩니다.
   * 
   * @param docs - 처리할 Document 객체 배열
   * @returns 번호가 매겨진 형식으로 결합된 문서 내용 문자열
   */
  private processDocs(docs: Document[]) {
    // 각 문서를 번호가 매겨진 형식으로 변환하고 결합
    return docs
      .map(
        // 문서 인덱스, 제목, 내용을 결합하여 형식화
        (_, index) =>
          `${index + 1}. ${docs[index].metadata.title} ${docs[index].pageContent}`,
      )
      .join('\n'); // 각 문서를 줄바꿈으로 구분하여 하나의 문자열로 결합
  }

  /**
   * 스트림 이벤트를 처리하고 이벤트 이미터를 통해 데이터를 전송하는 함수
   * 
   * 이 함수는 비동기 제너레이터로부터 생성되는 스트림 이벤트를 처리하고,
   * 이벤트 유형에 따라 적절한 데이터를 이벤트 이미터를 통해 클라이언트에 전송합니다.
   * 주로 검색 결과와 AI 응답을 실시간으로 스트리밍하는 데 사용됩니다.
   * 
   * @param stream - 처리할 스트림 이벤트의 비동기 제너레이터
   * @param emitter - 이벤트를 발생시키는 이벤트 이미터 객체
   */
  private async handleStream(
    stream: AsyncGenerator<StreamEvent, any, any>,
    emitter: eventEmitter,
  ) {
    // 스트림의 각 이벤트를 순차적으로 처리
    for await (const event of stream) {
      // 소스 검색 체인이 완료되었을 때 (검색 결과가 준비됨)
      if (
        event.event === 'on_chain_end' &&
        event.name === 'FinalSourceRetriever'
      ) {
        // 빈 문자열 (오타 또는 잔여 코드로 보임)
        ``;
        // 검색된 소스 데이터를 JSON 형식으로 변환하여 'data' 이벤트로 전송
        emitter.emit(
          'data',
          JSON.stringify({ type: 'sources', data: event.data.output }),
        );
      }
      
      // AI 응답이 생성되는 동안 (실시간 텍스트 청크)
      if (
        event.event === 'on_chain_stream' &&
        event.name === 'FinalResponseGenerator'
      ) {
        // 생성된 텍스트 청크를 JSON 형식으로 변환하여 'data' 이벤트로 전송
        emitter.emit(
          'data',
          JSON.stringify({ type: 'response', data: event.data.chunk }),
        );
      }
      
      // AI 응답 생성이 완료되었을 때
      if (
        event.event === 'on_chain_end' &&
        event.name === 'FinalResponseGenerator'
      ) {
        // 모든 처리가 완료되었음을 나타내는 'end' 이벤트 발생
        emitter.emit('end');
      }
    }
  }

  async searchAndAnswer(
    message: string,
    history: BaseMessage[],
    llm: BaseChatModel,
    embeddings: Embeddings,
    optimizationMode: 'speed' | 'balanced' | 'quality',
    fileIds: string[],
  ) {
    const emitter = new eventEmitter();

    const answeringChain = await this.createAnsweringChain(
      llm,
      fileIds,
      embeddings,
      optimizationMode,
    );

    const stream = answeringChain.streamEvents(
      {
        chat_history: history,
        query: message,
      },
      {
        version: 'v1',
      },
    );

    this.handleStream(stream, emitter);

    return emitter;
  }
}

export default MetaSearchAgent;
