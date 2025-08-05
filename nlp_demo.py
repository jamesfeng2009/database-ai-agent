#!/usr/bin/env python3
"""NLP模块功能演示."""

import asyncio
from src.sql_analyzer.nlp.processor import NLPProcessor


async def demonstrate_nlp_capabilities():
    """演示NLP模块的各种能力."""
    processor = NLPProcessor()
    
    print("=" * 60)
    print("数据库性能优化AI Agent - NLP模块功能演示")
    print("=" * 60)
    print()
    
    # 演示不同类型的用户输入
    test_cases = [
        {
            "input": "为什么我的用户表查询这么慢？",
            "description": "查询分析请求"
        },
        {
            "input": "帮我优化数据库性能，创建一些索引",
            "description": "优化请求"
        },
        {
            "input": "设置CPU使用率超过80%的告警",
            "description": "监控设置请求"
        },
        {
            "input": "什么是数据库索引？它是如何工作的？",
            "description": "知识查询"
        },
        {
            "input": "使用帮助，我不知道怎么操作",
            "description": "帮助请求"
        },
        {
            "input": "SELECT name, age FROM users WHERE id > 100",
            "description": "SQL语句分析"
        }
    ]
    
    for i, case in enumerate(test_cases, 1):
        print(f"示例 {i}: {case['description']}")
        print(f"用户输入: {case['input']}")
        print("-" * 40)
        
        # 提取意图和实体
        intent = await processor.extract_intent(case['input'])
        
        print(f"识别意图: {intent.intent_type.value}")
        print(f"置信度: {intent.confidence:.3f}")
        
        if intent.entities:
            print("提取的实体:")
            for entity_type, entities in intent.entities.items():
                print(f"  {entity_type}:")
                for entity in entities:
                    print(f"    - {entity['text']} (置信度: {entity['confidence']:.3f})")
        
        if intent.parameters:
            print("提取的参数:")
            for param, value in intent.parameters.items():
                print(f"  {param}: {value}")
        
        print()
    
    # 演示术语解释功能
    print("=" * 60)
    print("术语解释功能演示")
    print("=" * 60)
    
    terms_to_explain = ["索引", "慢查询", "执行计划", "优化", "监控"]
    explanations = await processor.explain_technical_terms(terms_to_explain)
    
    for term, explanation in explanations.items():
        print(f"{term}: {explanation}")
    
    print()
    
    # 演示统计信息
    print("=" * 60)
    print("NLP模块统计信息")
    print("=" * 60)
    
    stats = processor.get_processing_statistics()
    print(f"领域词典统计: {stats['domain_dict_stats']}")
    print(f"支持的意图类型: {stats['supported_intents']}")
    print(f"支持的实体类型: {stats['supported_entities']}")
    
    print()
    print("演示完成！")


if __name__ == "__main__":
    asyncio.run(demonstrate_nlp_capabilities())