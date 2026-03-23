#!/usr/bin/env python3
"""Test the decomposed extraction pipeline against ground truth."""

import json
import sys
from graph2svg.extractor import extract_graph
from graph2svg.model import GraphSpec


def score_extraction(result: GraphSpec, ground_truth: GraphSpec) -> dict:
    """Score an extraction result against ground truth."""
    gt_edges = {}
    for e in ground_truth.edges:
        key = (min(e.source, e.target), max(e.source, e.target), e.label)
        gt_edges[key] = e

    correct = 0
    wrong_direction = 0
    wrong_weight = 0
    phantom = 0
    missing = set(gt_edges.keys())

    for e in result.edges:
        key = (min(e.source, e.target), max(e.source, e.target), e.label)
        if key in gt_edges:
            missing.discard(key)
            gt = gt_edges[key]
            dir_ok = e.source == gt.source and e.target == gt.target
            weight_ok = e.weight == gt.weight

            if dir_ok and weight_ok:
                correct += 1
                print(f"  OK  {e.label}: {e.source}→{e.target} w={e.weight}")
            elif not dir_ok and not weight_ok:
                wrong_direction += 1
                wrong_weight += 1
                print(f"  BAD {e.label}: {e.source}→{e.target} w={e.weight} "
                      f"(expected {gt.source}→{gt.target} w={gt.weight})")
            elif not dir_ok:
                wrong_direction += 1
                print(f"  DIR {e.label}: {e.source}→{e.target} "
                      f"(expected {gt.source}→{gt.target})")
            else:
                wrong_weight += 1
                print(f"  WGT {e.label}: w={e.weight} (expected {gt.weight})")
        else:
            # Check if it matches by pair without label
            pair_key = (min(e.source, e.target), max(e.source, e.target))
            found_match = False
            for gk, ge in gt_edges.items():
                if (gk[0], gk[1]) == pair_key and gk in missing:
                    missing.discard(gk)
                    dir_ok = e.source == ge.source and e.target == ge.target
                    weight_ok = e.weight == ge.weight
                    label_note = f" (label mismatch: got {e.label}, expected {ge.label})"
                    if dir_ok and weight_ok:
                        correct += 1
                        print(f"  OK  {e.source}→{e.target} w={e.weight}{label_note}")
                    elif not dir_ok:
                        wrong_direction += 1
                        print(f"  DIR {e.source}→{e.target} (expected {ge.source}→{ge.target}){label_note}")
                    else:
                        wrong_weight += 1
                        print(f"  WGT {e.source}→{e.target} w={e.weight} (expected {ge.weight}){label_note}")
                    found_match = True
                    break
            if not found_match:
                phantom += 1
                print(f"  PHANTOM {e.label}: {e.source}→{e.target} w={e.weight}")

    for key in missing:
        gt = gt_edges[key]
        print(f"  MISSING {gt.label}: {gt.source}→{gt.target} w={gt.weight}")

    total = len(ground_truth.edges)
    return {
        "correct": correct,
        "total": total,
        "wrong_direction": wrong_direction,
        "wrong_weight": wrong_weight,
        "phantom": phantom,
        "missing": len(missing),
        "accuracy": correct / total if total > 0 else 0,
        "weight_format_ok": result.weight_format == ground_truth.weight_format,
    }


def main():
    tests = [
        ("examples/input1.png", "examples/input1_ground_truth.json"),
        ("examples/input2.jpg", "examples/input2_ground_truth.json"),
    ]

    for img_path, gt_path in tests:
        print(f"\n{'='*60}")
        print(f"Testing: {img_path}")
        print(f"{'='*60}")

        gt = GraphSpec.from_json(gt_path)

        try:
            result = extract_graph(
                image_path=img_path,
                model="gpt-4o",
                verbose=True,
                use_cache=True,
                pipeline="decomposed",
            )

            # Save result
            out_path = f"/tmp/decomposed_{img_path.split('/')[-1].split('.')[0]}.json"
            result.to_json(out_path)
            print(f"\nSaved result to {out_path}")

            print(f"\nNodes: {[n.name for n in result.nodes]}")
            print(f"Edges: {len(result.edges)}")
            print(f"Weight format: {result.weight_format}")
            print(f"\nScoring:")
            score = score_extraction(result, gt)
            print(f"\nResult: {score['correct']}/{score['total']} correct "
                  f"({score['accuracy']:.0%})")
            print(f"  Wrong direction: {score['wrong_direction']}")
            print(f"  Wrong weight: {score['wrong_weight']}")
            print(f"  Phantom edges: {score['phantom']}")
            print(f"  Missing edges: {score['missing']}")
            print(f"  Weight format correct: {score['weight_format_ok']}")

        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
