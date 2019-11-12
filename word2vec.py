import logging
import multiprocessing
import os.path
import sys
import jieba
import gensim
from gensim.models import Word2Vec
from gensim.models.word2vec import PathLineSentences

if __name__ == '__main__':
    
    # 日志信息输出
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))
    
    # check and process input arguments
    # if len(sys.argv) < 4:
    #     print(globals()['__doc__'] % locals())
    #     sys.exit(1)
    # input_dir, outp1, outp2 = sys.argv[1:4]
    
    input_dir = "C:\\Users\\18813\\Desktop\\project\\SinaMicrosoftNewsTitles.txt"
    outp1 = 'baike.model'
    outp2 = 'word2vec_format.bin'
    #sentences = Word2Vec.PathLineSentences("C:\\Users\\18813\\Desktop\\project\\SinaMicrosoftNewsTitles.txt")
    # 训练模型 
    # 输入语料目录:PathLineSentences(input_dir)
    # embedding size:256 共现窗口大小:10 去除出现次数5以下的词,多线程运行,迭代10次
    model = Word2Vec(PathLineSentences(input_dir),
                     size=256, window=10, min_count=5,
                     workers=multiprocessing.cpu_count(), iter=10)
    model.save(outp1)
    model.wv.save_word2vec_format(outp2, binary=True)
    wordVec = gensim.models.KeyedVectors.load_word2vec_format("word2Vec_format.bin", binary=True)

    # 运行命令:输入训练文件目录 python word2vec_model.py data baike.model baike.vector

